import os
import torch
import numpy as np
from typing import Union, List, Optional
from tqdm import tqdm

# 假设以下类和辅助函数已经在框架的其他地方定义好，此处直接引用以保持代码结构完整
# from your_framework import QwenConfig, ExecutionEnv, Policy, ValueHolder, Task
# from your_framework.layers import QwenMultimodalInput, QwenDecoderLayer, QwenOutputLayer
# from your_framework.utils import array_1d, array_2d, array_3d, timers, DUMMY_WEIGHT

class Qwen3VLLM:
    '''
    Qwen3VLLM 类：管理 Qwen3-VL 多模态模型的执行流程。
    
    架构映射：
    - Layer 0: Multimodal Input (Vision Encoder + Text Embed + Merger)
    - Layer 1...N: Text Decoder Layers (RoPE + SelfAttn + SwiGLU MLP)
    - Layer N+1: Output Layer (RMSNorm + LM Head)
    '''
    def __init__(self,
                 config: Union[str, object], # 假设为 QwenConfig
                 env: object,                # 假设为 ExecutionEnv
                 path: str,
                 policy: object):            # 假设为 Policy
        
        # 1. 配置与环境初始化
        if isinstance(config, str):
            # config = get_qwen_config(config) 
            pass # 实际代码中需实现配置加载
        self.config = config
        self.env = env
        self.path = path
        self.policy = policy
        self.num_gpu_batches = policy.num_gpu_batches

        # 2. 初始化网络层 (Layers Construction)
        layers = []
        
        # [Layer 0] 多模态输入层
        # 负责加载 model.visual 和 model.language_model.embed_tokens
        layers.append(QwenMultimodalInput(self.config, self.env, self.policy))
        
        # [Layer 1 ~ N] 文本解码层
        # 对应 model.language_model.layers
        for i in range(self.config.num_hidden_layers):
            layers.append(QwenDecoderLayer(self.config, self.env, self.policy, i))
        
        # [Layer N+1] 输出层
        # 对应 model.language_model.norm 和 lm_head
        layers.append(QwenOutputLayer(self.config, self.env, self.policy))
        
        self.layers = layers
        self.num_layers = len(layers)

        # 3. 确定激活值存储位置
        if self.policy.act_gpu_percent == 100:
            self.act_home = self.env.gpu
        elif self.policy.act_cpu_percent == 100:
            self.act_home = self.env.cpu
        elif self.policy.act_disk_percent == 100:
            self.act_home = self.env.disk
        else:
            raise NotImplementedError()

        # 4. 初始化 CUDA 流 (Streams)
        self.load_weight_stream = torch.cuda.Stream()
        self.load_cache_stream = torch.cuda.Stream()
        self.store_cache_stream = torch.cuda.Stream()

        # 5. 初始化中间张量缓冲区 (Intermediate Tensors)
        # 逻辑与 OptLM 保持一致
        num_layers, num_gpu_batches = self.num_layers, self.policy.num_gpu_batches

        self.cache_home = array_2d(num_layers, num_gpu_batches, ValueHolder)
        self.cache_read_buf = array_2d(num_layers, num_gpu_batches, ValueHolder)
        self.cache_write_buf = array_2d(num_layers, num_gpu_batches, ValueHolder)
        self.weight_read_buf = array_1d(num_layers, ValueHolder)
        self.attention_mask = array_1d(num_gpu_batches, ValueHolder)

        self.task = None
        self.init_all_weights()

    def set_task(self, task):
        self.task = task
        for l in self.layers:
            l.set_task(task)

    def init_weight(self, j):
        '''
        初始化第 j 层的权重。
        根据 Qwen3-VL 的结构映射文件路径。
        '''
        # 假设权重已转换为 numpy 格式存储在 path/model_name-np 目录下
        expanded_path = os.path.abspath(os.path.expanduser(
            os.path.join(self.path, f"{self.config.name}-np")))
        
        # 简单的权重存在性检查 (以 embed_tokens 为例)
        check_path = os.path.join(expanded_path, "model.language_model.embed_tokens.weight")
        if not os.path.exists(check_path) and "DUMMY_WEIGHT" not in check_path:
            # download_qwen_weights(self.config.name, self.path)
            pass

        # 构造 Layer j 对应的权重前缀或路径信息
        layer_info = {}
        if j == 0:
            layer_info['type'] = 'input'
            layer_info['visual_prefix'] = "model.visual"
            layer_info['text_embed_prefix'] = "model.language_model.embed_tokens"
        elif j == self.num_layers - 1:
            layer_info['type'] = 'output'
            layer_info['norm_prefix'] = "model.language_model.norm"
            layer_info['head_prefix'] = "lm_head"
        else:
            layer_info['type'] = 'decoder'
            # 注意：self.layers[1] 对应 text layer 0
            layer_idx = j - 1
            layer_info['prefix'] = f"model.language_model.layers.{layer_idx}"

        # 调用底层 Layer 的 init_weight，传入特定路径信息
        self.layers[j].init_weight(self.weight_home[j], expanded_path, layer_info)

    def load_weight(self, i, j, k, overlap=True):
        # 逻辑复用 OptLM
        if j == self.num_layers:
            j = 0
            i += 1
            if i == self.execute_gen_len:
                return

        if overlap:
            with torch.cuda.stream(self.load_weight_stream):
                self.layers[j].load_weight(self.weight_home[j], self.weight_read_buf[j], k)
        else:
            self.layers[j].load_weight(self.weight_home[j], self.weight_read_buf[j], k)

    def delete_weight(self, j, k):
        # 逻辑复用 OptLM
        if k == 0:
            for x in self.weight_home[j].pop():
                if isinstance(x, ValueHolder):
                    for y in x.pop():
                        y.delete()
                else:
                    x.delete()

    def init_cache(self, j, k):
        # 初始化 KV Cache。
        # 注意：Layer 0 (Input) 通常不需要 KV Cache，底层实现应直接返回
        self.layers[j].init_cache_one_gpu_batch(self.cache_home[j][k])

    def load_cache(self, i, j, k, overlap=True):
        # 逻辑复用 OptLM
        if i == 0:  # prefill 阶段无需加载 cache
            return
        if k == self.num_gpu_batches:
            k = 0
            j += 1
        if j == self.num_layers:
            j = 0
            i += 1
            if i == self.execute_gen_len:
                return

        if overlap:
            with torch.cuda.stream(self.load_cache_stream):
                self.layers[j].load_cache(self.cache_home[j][k], self.cache_read_buf[j][k], i)
        else:
            self.layers[j].load_cache(self.cache_home[j][k], self.cache_read_buf[j][k], i)

    def store_cache(self, i, j, k, overlap=True):
        # 逻辑复用 OptLM
        if k == -1:
            k = self.num_gpu_batches - 1
            j -= 1
        if j == -1:
            j = self.num_layers - 1
            i -= 1
            if i == -1:
                return
        if i == self.task.gen_len - 1:
            self.cache_write_buf[j][k].pop()
            return

        if overlap:
            with torch.cuda.stream(self.store_cache_stream):
                self.layers[j].store_cache(self.cache_home[j][k], self.cache_write_buf[j][k], i)
        else:
            self.layers[j].store_cache(self.cache_home[j][k], self.cache_write_buf[j][k], i)

    def delete_cache(self, j, k):
        v = self.cache_home[j][k].pop()
        if v:
            for x in v:
                x.delete()

    def load_hidden(self, i, j, k):
        # 管理 hidden states 的加载
        if k == self.num_gpu_batches:
            k = 0
            j += 1
        if j == self.num_layers:
            j = 0
            i += 1
            if i == self.execute_gen_len:
                return

        dst = self.layers[j].compute
        if j == 0:
            # [Layer 0 特殊处理] 
            # 这一层需要加载 input_ids。对于多模态模型，pixel_values 通常存储在 Task 对象中，
            # 由 Layer 0 内部直接访问，不需要在这里通过 hidden buffer 传递。
            gpu_batch_size = self.policy.gpu_batch_size
            left, right = k * gpu_batch_size, (k + 1) * gpu_batch_size
            
            if i == 0:  # prefill: load full prompt
                val = dst.allocate((gpu_batch_size, self.task.prompt_len), np.int32)
                val.load_from_np(self.output_ids[left:right, :self.task.prompt_len])
            else:  # decoding: load last token
                pos = self.task.prompt_len + i
                val = dst.allocate((gpu_batch_size, 1), np.int32)
                val.load_from_np(self.output_ids[left:right, pos-1:pos])
        else:
            # 从上一层加载 hidden state
            val = self.hidden[i][j-1][k].pop().move(dst)
        
        self.hidden[i][j][k].store(val)

    def store_hidden(self, i, j, k):
        # 管理 hidden states 的存储
        if k == -1:
            k = self.num_gpu_batches - 1
            j -= 1
        if j == -1:
            j = self.num_layers - 1
            i -= 1
            if i == -1:
                return

        if j == self.num_layers - 1:  # 最后一层输出处理
            gpu_batch_size = self.policy.gpu_batch_size
            left, right = k * gpu_batch_size, (k + 1) * gpu_batch_size
            ids = self.hidden[i][j][k].pop().data.detach().cpu().numpy()
            pos = self.task.prompt_len + i
            
            if self.task.stop:
                stopped = self.stopped[left:right]
                self.output_ids[left:right, pos:pos+1] = np.where(
                    stopped, self.config.pad_token_id, ids)
                stopped[:] = np.logical_or(stopped, ids == self.task.stop)
            else:
                self.output_ids[left:right, pos:pos+1] = ids
        else:
            # 将中间结果移动到 act_home (CPU/GPU/Disk)
            x = self.hidden[i][j][k]
            if x.val:
                x.val = x.val.move(self.act_home)

    def compute_layer(self, i, j, k):
        # 执行计算
        # 注意：Qwen 的 Attention Mask 和 RoPE 计算由 Layer 内部处理
        self.layers[j].forward(self.hidden[i][j][k], self.cache_read_buf[j][k],
            self.weight_read_buf[j], self.attention_mask[k],
            self.cache_write_buf[j][k], i, k)

    def sync(self):
        self.env.disk.synchronize()
        torch.cuda.synchronize()

    def init_all_weights(self):
        self.weight_home = array_1d(self.num_layers, ValueHolder)
        for j in range(self.num_layers):
            self.init_weight(j)

    def delete_all_weights(self):
        for j in range(self.num_layers):
            self.delete_weight(j, 0)

    def update_attention_mask(self, i, k):
        # Qwen3-VL 的掩码逻辑。
        # 如果 i > 0 (Decoding)，通常通过 KV Cache 的更新机制隐式处理，或者只需要简单的 extend
        if i > 0:
            mask = self.attention_mask[k]
            if mask.val is not None:
                # 假设底层支持 extend_attention_mask 操作
                mask.val = mask.val.device.extend_attention_mask(mask.val, [True])
            return

        # Prefill 阶段构建 Mask
        gpu_batch_size = self.policy.gpu_batch_size
        left = k * gpu_batch_size
        right = left + gpu_batch_size
        input_ids = self.output_ids[left:right, :self.task.prompt_len]

        attention_compute = (self.env.cpu if self.policy.cpu_cache_compute
            else self.env.gpu)
        val = attention_compute.allocate(
            (self.policy.gpu_batch_size, self.task.prompt_len), bool)
        
        # 注意：Qwen3 可能使用特殊的 pad_token_id
        val.load_from_np((input_ids != self.config.pad_token_id))
        self.attention_mask[k].store(val)

    def generate(self,
                 inputs: Union[np.array, List[List[int]]],
                 pixel_values: Optional[Union[np.array, List]] = None, # 新增：支持 Qwen 图像输入
                 image_grid_thw: Optional[Union[np.array, List]] = None, # 新增：支持 Qwen3VL grid参数
                 max_new_tokens: int = 32,
                 do_sample: bool = False,
                 temperature: float = 1.0,
                 stop: Optional[int] = None,
                 debug_mode: Optional[str] = None,
                 cut_gen_len: Optional[int] = None,
                 verbose: int = 0):
        
        # 构建 Task 对象，传入多模态数据
        # 假设 Task 类已经扩展以支持 pixel_values
        task = Task(
            inputs=inputs,
            pixel_values=pixel_values,       # Qwen3VL 特有
            image_grid_thw=image_grid_thw,   # Qwen3VL 特有
            prompt_len=len(inputs[0]),
            gen_len=max_new_tokens,
            cut_gen_len=cut_gen_len,
            do_sample=do_sample,
            temperature=temperature,
            stop=stop,
        )
        
        num_layers = self.num_layers
        num_gpu_batches = self.num_gpu_batches
        gpu_batch_size = self.policy.gpu_batch_size
        overlap = self.policy.overlap
        prompt_len, gen_len = task.prompt_len, task.gen_len
        self.execute_gen_len = task.cut_gen_len if task.cut_gen_len else task.gen_len

        # Output token ids
        self.output_ids = np.full((len(task.inputs), prompt_len + gen_len),
            self.config.pad_token_id, dtype=np.int32)
        self.stopped = np.zeros((len(task.inputs), 1), dtype=bool)
        self.output_ids[:, :prompt_len] = np.asarray(task.inputs)
        assert gpu_batch_size * num_gpu_batches == len(task.inputs)

        # 清理缓冲区
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
        
        # 初始化 Hidden Buffer: [gen_len, num_layers, num_gpu_batches]
        self.hidden = array_3d(gen_len, num_layers, num_gpu_batches, ValueHolder)

        # Init cache & Task
        self.set_task(task)
        for j in range(num_layers):
            for k in range(num_gpu_batches):
                self.init_cache(j, k)
        
        if self.policy.cpu_cache_compute:
            self.env.cpu.init_attention_compute_workspace(self.config, self.task, self.policy)

        # 执行生成循环 (Pipeline Loop)
        if debug_mode is None:
            if not overlap:
                self.generation_loop_normal()
            else:
                if num_gpu_batches == 1:
                    self.generation_loop_overlap_single_batch()
                else:
                    self.generation_loop_overlap_multi_batch()
        elif debug_mode == "fewer_batch":
            if num_gpu_batches == 1:
                self.generation_loop_debug_single_batch()
            else:
                self.generation_loop_debug_multi_batch()
        elif debug_mode == "breakdown":
            self.generation_loop_debug_normal()
        else:
            raise ValueError(f"Invalid debug mode: {debug_mode}")

        # 清理 Cache
        for j in range(num_layers):
            for k in range(num_gpu_batches):
                self.delete_cache(j, k)
        if self.policy.cpu_cache_compute:
            self.env.cpu.del_attention_compute_workspace()

        return self.output_ids

    # --- 以下循环逻辑与 OptLM 保持一致，无需更改 ---
    # 因为所有模型特定的差异已被封装在 Layer 对象和 init_weight 中
    
    def generation_loop_normal(self):
        for i in range(self.execute_gen_len):
            timers("generate").start()
            for k in range(self.num_gpu_batches):
                self.update_attention_mask(i, k)
            for j in range(self.num_layers):
                for k in range(self.num_gpu_batches):
                    self.load_weight(i, j, k, overlap=False)

                for k in range(self.num_gpu_batches):
                    self.load_cache(i, j, k, overlap=False)
                    self.load_hidden(i, j, k)
                    self.compute_layer(i, j, k)
                    self.store_hidden(i, j, k)
                    self.store_cache(i, j, k, overlap=False)
            timers("generate").stop()

    def generation_loop_debug_normal(self):
        execute_num_batches = 20
        batch_ct = 0
        pbar = tqdm(total=execute_num_batches)
        timers("prefill_total").reset()
        timers("decoding_gpu_batch").reset()
        # ... (定时器重置省略) ...
        load_weight_timer = timers("load_weight")

        for i in range(self.execute_gen_len):
            if i == 0:
                timers("prefill_total").start()
                # ... (定时器变量赋值省略) ...
            else:
                # ... (定时器变量赋值省略) ...
                pass
            
            # 简化代码，逻辑同 OptLM
            for k in range(self.num_gpu_batches):
                self.update_attention_mask(i, k)

            for j in range(self.num_layers):
                if i > 0: timers("decoding_gpu_batch").start()

                load_weight_timer.start(self.sync)
                for k in range(self.num_gpu_batches):
                    self.load_weight(i, j, k)
                load_weight_timer.stop(self.sync)

                for k in range(self.num_gpu_batches):
                    self.load_cache(i, j, k)
                    self.load_hidden(i, j, k)
                    self.compute_layer(i, j, k)
                    self.store_hidden(i, j, k)
                    self.store_cache(i, j, k)

                if i > 0:
                    timers("decoding_gpu_batch").stop()
                    pbar.update(1)
                    batch_ct += 1
                if batch_ct >= execute_num_batches: break
            if batch_ct >= execute_num_batches: break
            if i == 0: timers("prefill_total").stop(self.sync)
        
        # ... (Debug 打印省略) ...

    def generation_loop_overlap_single_batch(self):
        # Prologue
        for k in range(self.num_gpu_batches):
            self.load_weight(0, 0, k)
        self.sync()

        # Generate
        for i in range(self.execute_gen_len):
            timers("generate").start()
            self.update_attention_mask(i, 0)
            for j in range(self.num_layers):
                self.load_weight(i, j+1, 0)
                self.load_cache(i, j+1, 0)
                self.load_hidden(i, j, 0)
                self.compute_layer(i, j, 0)
                self.store_cache(i, j-1, 0)
                self.store_hidden(i, j, 0)
                self.sync()
            timers("generate").stop()

            if self.task.stop and np.all(self.stopped):
                break

    def generation_loop_overlap_multi_batch(self):
        # Prologue
        for k in range(self.num_gpu_batches):
            self.load_weight(0, 0, k)
        self.load_hidden(0, 0, 0)
        self.sync()

        # Generate
        for i in range(self.execute_gen_len):
            timers("generate").start()
            for k in range(self.num_gpu_batches):
                self.update_attention_mask(i, k)
            for j in range(self.num_layers):
                for k in range(self.num_gpu_batches):
                    self.load_weight(i, j+1, k)
                    self.load_cache(i, j, k+1)
                    self.store_hidden(i, j, k-1)
                    self.load_hidden(i, j, k+1)
                    self.compute_layer(i, j, k)
                    self.store_cache(i, j, k-1)
                    self.sync()
            timers("generate").stop()

        # Epilogue
        self.store_hidden(
            self.execute_gen_len-1, self.num_layers-1, self.num_gpu_batches-1)

    def generation_loop_debug_single_batch(self):
        # 复用 OptLM 逻辑
        self.generation_loop_normal() # 简化起见调用 normal

    def generation_loop_debug_multi_batch(self):
        # 复用 OptLM 逻辑
        self.generation_loop_normal() # 简化起见调用 normal

    def __del__(self):
        self.delete_all_weights()