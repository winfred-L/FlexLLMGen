from abc import ABC, abstractmethod
from typing import Union, List, Optional, Tuple
import numpy as np
import torch
import os

from transformers.feature_extraction_utils import BatchFeature

from flexllmgen.timer import timers
from flexllmgen.utils import (Task, VisionTask, ExecutionEnv, MonitoredBuffer, ValueHolder)
from flexllmgen.models.base_config import BaseConfig
from flexllmgen.policy import Policy

class BaseFlexLM(ABC):
    '''
    管理模型执行流程的基类，已经实现了以下方法：
    1. __init__
    2. set_task
    3. init_all_weights, delete_all_weights, init_weight, delete_weight, load_weight
    4. init_cache, delete_cache, load_cache, store_cache
    5. load_hidden, store_hidden
    6. compute_layer
    7. update_attention_mask
    8. sync
    9. generate

    需要子类实现的抽象方法：
    1. get_model_config  设置模型配置
    2. init_model_layers 模型层初始化构建
    3. get_task          设置模型推理任务
    4. generation_loop_normal
       generation_loop_debug_normal
       generation_loop_overlap_single_batch
       generation_loop_overlap_multi_batch
       generation_loop_debug_single_batch
       generation_loop_debug_multi_batch
    '''

    def __init__(self,
                 name: str,
                 env: ExecutionEnv,
                 path: str, # 模型权重文件所在目录
                 policy: Policy):
        self.name = name
        self.env = env
        self.policy = policy
        self.config = self.get_model_config() # abstractmethod
        self.num_gpu_batches = policy.num_gpu_batches

        # --- 初始化网络层 ---
        self.layers = self.init_model_layers() # abstractmethod
        self.num_layers = len(self.layers)

        # --- 确定激活值(Activation)的存储位置 ---
        # 根据策略决定层与层之间传递的中间结果（Hidden States）存储在哪里
        if self.policy.act_gpu_percent == 100:
            self.act_home = self.env.gpu
        elif self.policy.act_cpu_percent == 100:
            self.act_home = self.env.cpu
        elif self.policy.act_disk_percent == 100:
            self.act_home = self.env.disk
        else:
            raise NotImplementedError()

        # --- 初始化 CUDA 流 ---
        # 使用独立的 CUDA 流来实现计算与数据传输的并行（Overlap）
        self.load_weight_stream = torch.cuda.Stream()
        self.load_cache_stream = torch.cuda.Stream()
        self.store_cache_stream = torch.cuda.Stream()

        # --- 初始化中间张量缓冲区 (Intermediate Tensors) ---
        # The following buffers store values used
        # for the i-th token, j-th layer, k-th gpu batch.
        num_layers, num_gpu_batches = self.num_layers, self.policy.num_gpu_batches
        
        # cache[j][k]
        # cache_home: 存储所有层、所有 batch 的完整 KV Cache
        self.cache_home = MonitoredBuffer(
            shape=(num_layers, num_gpu_batches),
            name="cache_home",
        )
        # cache_read_buf: 当前计算需要的 KV Cache（通常在 GPU 上）
        self.cache_read_buf = MonitoredBuffer(
            shape=(num_layers, num_gpu_batches),
            name="cache_read_buf",
        )
        # cache_write_buf: 计算产生的新 KV Cache，等待写回
        self.cache_write_buf = MonitoredBuffer(
            shape=(num_layers, num_gpu_batches),
            name="cache_write_buf",
        )

        # weight[j]
        # weight_read_buf: 当前计算层需要的权重（预取到 GPU）
        self.weight_read_buf = MonitoredBuffer(
            shape=(num_layers,),
            name="weight_read_buf",
        )
        
        # attention_mask[k]
        # attention_mask: 存储 Attention Mask
        self.attention_mask = MonitoredBuffer(
            shape=(num_gpu_batches,),
            name="attention_mask",
        )

        self.task = None
        self.weight_file_path = os.path.abspath(os.path.expanduser(
            os.path.join(path, f"{self.config.name}-np")))
        self.init_all_weights()

    def __del__(self):
        self.delete_all_weights()

    # --- Weight 管理 ---
    def init_all_weights(self):
        self.weight_home = MonitoredBuffer(
            shape=(self.num_layers,),
            name="weight_home",
        )
        for j in range(self.num_layers):
            self.init_weight(j)

    def delete_all_weights(self):
        for j in range(self.num_layers):
            self.delete_weight(j, 0)

    def init_weight(self, j):
        # 去掉了路径正确性检查
        self.layers[j].init_weight(self.weight_home[j], self.weight_file_path)

    def delete_weight(self, j, k):
        if k == 0:
            for x in self.weight_home[j].pop():
                if isinstance(x, ValueHolder):
                    for y in x.pop(): y.delete()
                else:
                    x.delete()

    def load_weight(self, i, j, k, overlap=True):
        # Handle corner cases
        if j == self.num_layers:
            j = 0
            i += 1
            if i == self.execute_gen_len:
                return

        # Load from weight_home to weight_read_buf
        if overlap:
            with torch.cuda.stream(self.load_weight_stream):
                self.layers[j].load_weight(self.weight_home[j], self.weight_read_buf[j], k)
        else:
            self.layers[j].load_weight(self.weight_home[j], self.weight_read_buf[j], k)

    # --- Cache 管理 ---
    def init_cache(self, j, k):
        self.layers[j].init_cache_one_gpu_batch(self.cache_home[j][k])

    def delete_cache(self, j, k):
        v = self.cache_home[j][k].pop()
        if v:
            for x in v: x.delete()

    def load_cache(self, i, j, k, overlap=True):
        # Handle corner cases
        if i == 0:  # prefill, no cache
            return
        if k == self.num_gpu_batches:
            k = 0
            j += 1
        if j == self.num_layers:
            j = 0
            i += 1
            if i == self.execute_gen_len:
                return

        # Load from cache_home to cache_read_buf
        if overlap:
            with torch.cuda.stream(self.load_cache_stream):
                self.layers[j].load_cache(self.cache_home[j][k], self.cache_read_buf[j][k], i)
        else:
            self.layers[j].load_cache(self.cache_home[j][k], self.cache_read_buf[j][k], i)

    def store_cache(self, i, j, k, overlap=True):
        # Handle corner cases
        if k == -1:
            k = self.num_gpu_batches - 1
            j -= 1
        if j == -1:
            j = self.num_layers - 1
            i -= 1
            if i == -1:
                return
        if i == self.task.gen_len - 1:  # last token, no need to store cache
            self.cache_write_buf[j][k].pop()
            return

        # Store cache_write_buf to cache_home
        # Delete cache_write_buf
        if overlap:
            with torch.cuda.stream(self.store_cache_stream):
                self.layers[j].store_cache(self.cache_home[j][k], self.cache_write_buf[j][k], i)
        else:
            self.layers[j].store_cache(self.cache_home[j][k], self.cache_write_buf[j][k], i)

    # --- Hidden States 管理 ---
    def load_hidden(self, i, j, k):
        # Handle corner cases
        if k == self.num_gpu_batches:
            k = 0
            j += 1
        if j == self.num_layers:
            j = 0
            i += 1
            if i == self.execute_gen_len:
                return

        # Load to hidden states buffers
        dst = self.layers[j].compute
        if j == 0:
            gpu_batch_size = self.policy.gpu_batch_size
            left, right = k * gpu_batch_size, (k + 1) * gpu_batch_size
            if i == 0:  # load from the input ids
                val = dst.allocate((gpu_batch_size, self.task.prompt_len), np.int32)
                val.load_from_np(self.output_ids[left:right, :self.task.prompt_len])
            else:  # load from the last generated token
                pos = self.task.prompt_len + i
                val = dst.allocate((gpu_batch_size, 1), np.int32)
                val.load_from_np(self.output_ids[left:right, pos-1:pos])
        else:  # load from the last layer
            val = self.hidden[i][j-1][k].pop().move(dst)
        self.hidden[i][j][k].store(val)

    def store_hidden(self, i, j, k):
        # Handle corner cases
        if k == -1:
            k = self.num_gpu_batches - 1
            j -= 1
        if j == -1:
            j = self.num_layers - 1
            i -= 1
            if i == -1:
                return

        # Store to hidden states buffers
        if j == self.num_layers - 1:  # store to output
            gpu_batch_size = self.policy.gpu_batch_size
            left, right = k * gpu_batch_size, (k + 1) * gpu_batch_size
            ids = self.hidden[i][j][k].pop().data.detach().cpu().numpy()
            pos = self.task.prompt_len + i
            if self.task.stop:
                stopped = self.stopped[left:right]
                self.output_ids[left:right, pos:pos+1] = np.where(
                    stopped, self.config.pad_token_id, ids)
                if self.task.stop is int:
                    flag = ids == self.task.stop
                else:  # tuple
                    flag = False
                    for stop_id in self.task.stop:
                        flag |= (ids == stop_id)
                stopped[:] = np.logical_or(stopped, flag)
            else:
                self.output_ids[left:right, pos:pos+1] = ids
        else:  # move to home
            x = self.hidden[i][j][k]
            if x.val:  # x may already be moved due to overlapping
                x.val = x.val.move(self.act_home)

    # --- 层计算 ---
    def compute_layer(self, i, j, k):
        # Update the hidden in place
        # Clear the weight_read_buf if it is the last gpu batch
        # Clear the cache_read_buf
        # Run layer computation
        self.layers[j].forward(
            self.hidden[i][j][k], 
            self.cache_read_buf[j][k],
            self.weight_read_buf[j], 
            self.attention_mask[k],
            self.cache_write_buf[j][k], 
            i, k
        )
            
    # --- 辅助函数 ---
    def update_attention_mask(self, i, k):
        if i > 0:
            mask = self.attention_mask[k]
            assert mask.val is not None
            mask.val = mask.val.device.extend_attention_mask(mask.val, [True])
            return

        gpu_batch_size = self.policy.gpu_batch_size
        left = k * gpu_batch_size
        right = left + gpu_batch_size
        input_ids = self.output_ids[left:right, :self.task.prompt_len]

        attention_compute = (self.env.cpu if self.policy.cpu_cache_compute
            else self.env.gpu)
        val = attention_compute.allocate(
            (self.policy.gpu_batch_size, self.task.prompt_len), bool)
        val.load_from_np((input_ids != self.config.pad_token_id))
        self.attention_mask[k].store(val)

    def sync(self):
        self.env.disk.synchronize()
        torch.cuda.synchronize()

    def set_task(self, task):
        self.task = task
        for l in self.layers:
            l.set_task(task)

    # --- 生成主入口 ---
    def generate(self,
                 inputs: Union[np.array, List[List[int]], BatchFeature],
                 max_new_tokens: int = 32,
                 do_sample: bool = False,
                 temperature: float = 1.0,
                 stop: Optional[int] = None,
                 debug_mode: Optional[str] = None,
                 cut_gen_len: Optional[int] = None,
                 verbose: int = 0):

        task = self.get_task(inputs, max_new_tokens, cut_gen_len, do_sample, temperature, stop) # abstractmethod
        self.set_task(task)

        # Output token ids
        prompt_len, gen_len = task.prompt_len, task.gen_len
        self.output_ids = np.full((len(task.input_ids), prompt_len + gen_len),
            self.config.pad_token_id, dtype=np.int32)
        self.stopped = np.zeros((len(task.input_ids), 1), dtype=bool)
        self.output_ids[:, :prompt_len] = np.asarray(task.input_ids)

        # Intermediate tensors
        # The following buffers store values used
        # for the i-th token, j-th layer, k-th gpu batch.
        num_layers, num_gpu_batches = self.num_layers, self.policy.num_gpu_batches
        for j in range(num_layers):
            for k in range(num_gpu_batches):
                self.cache_home[j][k].clear()
                self.cache_read_buf[j][k].clear()
                self.cache_write_buf[j][k].clear()
        for j in range(num_layers):
            self.weight_read_buf[j].clear()
        for k in range(num_gpu_batches):
            self.attention_mask[k].clear()
        self.hidden = MonitoredBuffer(
            shape=(gen_len, num_layers, num_gpu_batches),
            name="hidden",
        )
        
        # Init cache
        for j in range(num_layers):
            for k in range(num_gpu_batches):
                self.init_cache(j, k)
        if self.policy.cpu_cache_compute:
            self.env.cpu.init_attention_compute_workspace(self.config, self.task, self.policy)
        
        # Generate
        self.execute_gen_len = task.cut_gen_len if task.cut_gen_len else task.gen_len
        if debug_mode is None:
            if not self.policy.overlap:
                # No overlap, easy to understand, suitable for debugging
                self.generation_loop_normal() # abstractmethod
            else:
                # Overlap I/O and compute
                if num_gpu_batches == 1:
                    self.generation_loop_overlap_single_batch() # abstractmethod
                else:
                    self.generation_loop_overlap_multi_batch() # abstractmethod
        elif debug_mode == "fewer_batch":
            # Run fewer layeres and batches for debugging
            if num_gpu_batches == 1:
                self.generation_loop_debug_single_batch() # abstractmethod
            else:
                self.generation_loop_debug_multi_batch() # abstractmethod
        elif debug_mode == "breakdown":
            if not self.policy.overlap:
                # No overlap, fewer batches, execution time breakdown
                self.generation_loop_debug_normal() # abstractmethod
            else:
                # Overlap I/O and compute
                if num_gpu_batches == 1:
                    self.generation_loop_debug_overlap_single_batch() # abstractmethod
                else:
                    self.generation_loop_debug_overlap_multi_batch() # abstractmethod
        else:
            raise ValueError("Invalid debug mode: {debug_mode}")

        # Delete cache
        for j in range(num_layers):
            for k in range(num_gpu_batches):
                self.delete_cache(j, k)
        if self.policy.cpu_cache_compute:
            self.env.cpu.del_attention_compute_workspace()

        # print memory stats
        self.print_memory_stats()

        return self.output_ids
    

    def print_memory_stats(self):

        def format_size(size_bytes):
            if size_bytes == 0:
                return "0.00  B"
            units = [" B", "KB", "MB", "GB", "TB"]
            i = 0
            size = float(size_bytes)
            while size >= 1024 and i < len(units) - 1:
                size /= 1024
                i += 1
            return f"{size:.2f} {units[i]}"
        
        print("\n"+"="*60+"\n"+f"{'INFERENCE MEMORY REPORT':^60}\n"+"="*60)
        buffer_list = [
            self.weight_home,
            self.weight_read_buf,
            self.attention_mask,
            self.cache_home,
            self.cache_read_buf,
            self.cache_write_buf,
            self.hidden,
        ]
        for buffer in buffer_list:
            formatted_size = format_size(buffer.peak_bytes)
            print(f"{buffer.name:<30} Peak Memory: {formatted_size:>12}")
        print("="*60+"\n")


    @abstractmethod
    def get_model_config(self) -> BaseConfig:
        pass

    @abstractmethod
    def init_model_layers(self) -> List:
        pass

    @abstractmethod
    def get_task(self, inputs, max_new_tokens, cut_gen_len, do_sample, temperature, stop) -> Union[Task, VisionTask]:
        pass

    @abstractmethod
    def generation_loop_normal(self):
        pass

    @abstractmethod
    def generation_loop_debug_normal(self):
        pass

    @abstractmethod
    def generation_loop_overlap_single_batch(self):
        pass

    @abstractmethod
    def generation_loop_overlap_multi_batch(self):
        pass

    @abstractmethod
    def generation_loop_debug_single_batch(self):
        pass

    @abstractmethod
    def generation_loop_debug_multi_batch(self):
        pass

    @abstractmethod
    def generation_loop_debug_overlap_single_batch(self):
        pass

    @abstractmethod
    def generation_loop_debug_overlap_multi_batch(self):
        pass
