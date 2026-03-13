import argparse
import dataclasses
from attr import define, field
from attr.setters import frozen
import functools
import gc
import math
import os
from typing import Tuple, Union, Optional, Any, Sequence, List

import numpy as np
import torch


KB = 1 << 10
MB = 1 << 20
GB = 1 << 30
T = 1e12


@dataclasses.dataclass(frozen=True)
class Task:
    """A generation task."""
    input_ids: Union[np.array, List[List[int]]]
    prompt_len: int
    gen_len: int
    cut_gen_len: Optional[int]

    do_sample: bool
    temperature: float
    stop: Union[int, Tuple[int], None]


@dataclasses.dataclass(frozen=True)
class VideoInfo:
    T_len: int
    H_len: int
    W_len: int
    index_ranges: list[tuple[int, int]]
    
    @property
    def total_len(self) -> int:
        return self.T_len * self.H_len * self.W_len
    
    # @property
    # def total_len(self) -> int:
    #     total_l = 0
    #     for start, end in self.index_ranges:
    #         total_l += end - start + 1
    #     return total_l


@dataclasses.dataclass(frozen=True)
class VisionTask(Task):
    attention_mask: torch.Tensor
    # pixel_values: torch.Tensor
    pixel_values_videos: torch.Tensor
    # image_grid_thw: torch.Tensor
    video_grid_thw: torch.Tensor
    second_per_grid_ts: torch.Tensor
    video_info: VideoInfo



@dataclasses.dataclass(frozen=True)
class ExecutionEnv:
    """Hardware environment."""
    gpu: Any = None
    cpu: Any = None
    disk: Any = None
    mixed: Any = None

    @classmethod
    def create(cls, offload_dir):
        # fix recursive import
        from flexllmgen.pytorch_backend import TorchDevice, TorchDisk, TorchMixedDevice
        gpu = TorchDevice("cuda:0")
        cpu = TorchDevice("cpu")
        disk = TorchDisk(offload_dir)
        return cls(gpu=gpu, cpu=cpu, disk=disk, mixed=TorchMixedDevice([gpu, cpu, disk]))

    def close_copy_threads(self):
        self.disk.close_copy_threads()


@dataclasses.dataclass(frozen=True)
class BenchmarkResult:
    """Benchmark results."""
    prefill_latency: float
    prefill_throughput: float
    decode_latency: float
    decode_throughput: float
    total_latency: float
    total_throughput: float


# 因为 numpy 不支持 bfloat16，所以这里映射到 uint16
np_dtype_to_torch_dtype = {
    np.uint16: torch.bfloat16,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int32: torch.int32,
    np.int64: torch.int64,
    bool: torch.bool,
}

torch_dtype_to_np_dtype = {
    torch.bfloat16: np.uint16,
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.uint8: np.uint8,
    torch.int8: np.int8,
    torch.int32: np.int32,
    torch.int64: np.int64,
    torch.bool: bool,
}

torch_dtype_to_num_bytes = {
    torch.bfloat16: 2,
    torch.float16: 2,
    torch.float32: 4,
    torch.int8: 1,
    torch.uint8: 1,
    torch.int32: 4,
    torch.int64: 8,
    torch.bool: 1,
}


def piecewise_linear_func(xs, ys):
    """Return a function created by linear inerpolation."""
    indices = np.argsort(xs)
    xs = [xs[i] for i in indices]
    ys = [ys[i] for i in indices]

    # pad left and right
    k = 1e5
    delta_x_left = xs[0] - xs[1]
    delta_y_left = ys[0] - ys[1]
    delta_x_right = xs[-1] - xs[-2]
    delta_y_right = ys[-1] - ys[-2]

    xs = [xs[0] + delta_x_left * k] + xs + [xs[-1] + delta_x_right * k]
    ys = [ys[0] + delta_y_left * k] + ys + [ys[-1] + delta_y_right * k]

    return functools.partial(piecewise_linear_func_ret_func, xs, ys)


def piecewise_linear_func_ret_func(xs, ys, x):
    assert x >= xs[0] and x <= xs[-1]
    return np.interp(x, xs, ys)


def sample_from_range(n, k):
    assert n >= 1

    if k == -1:
        ret = [1]
        while ret[-1] * 2 < n:
            ret.append(ret[-1] * 2)
        return ret
    else:
        if k == 1: return [1]
        step = (n - 1) // (k - 1)
        return list(range(1, n + 1, step))


def cpu_mem_stats():
    objects = gc.get_objects()
    tensors = [obj for obj in objects if torch.is_tensor(obj) and not obj.is_cuda]

    total_numel = 0
    total_mem = 0
    visited_data = set()
    for tensor in tensors:
        # a data_ptr indicates a memory block allocated
        data_ptr = tensor.storage().data_ptr()
        if data_ptr in visited_data:
            continue
        visited_data.add(data_ptr)

        numel = tensor.storage().nbytes() # tensor.numel()
        total_numel += numel
        element_size = tensor.storage().element_size()
        mem = numel * element_size
        total_mem += mem
        '''
        以上原始实现中，如果有一个较大的tensor A，以及一个较小的切片B=A[0:1]，
        两个tensor A和B共享同一块内存区域。循环先遍历到B时，B.numel()只统计到
        较小部分，而完整的存储空间在后遍历到A时会跳过，存在BUG。
        '''

    return total_mem


def torch_mem_stats():
    objects = gc.get_objects()
    tensors = [obj for obj in objects if torch.is_tensor(obj) and obj.is_cuda]

    total_numel = 0
    total_mem = 0
    visited_data = set()
    for tensor in tensors:
        # a data_ptr indicates a memory block allocated
        data_ptr = tensor.storage().data_ptr()
        if data_ptr in visited_data:
            continue
        visited_data.add(data_ptr)

        print(tensor.shape, tensor.data_ptr())

        numel = tensor.storage().nbytes() # tensor.numel()
        total_numel += numel
        element_size = tensor.storage().element_size()
        mem = numel * element_size
        total_mem += mem

    return total_mem


class ValueHolder:
    def __init__(self):
        self.val = None

    def store(self, val):
        assert self.val is None
        self.val = val

    def pop(self):
        ret = self.val
        self.val = None
        return ret

    def clear(self):
        self.val = None


def array_1d(a, cls):
    return [cls() for _ in range(a)]


def array_2d(a, b, cls):
    return [[cls() for _ in range(b)] for _ in range(a)]


def array_3d(a, b, c, cls):
    return [[[cls() for _ in range(c)] for _ in range(b)] for _ in range(a)]


def array_4d(a, b, c, d, cls):
    return [[[[cls() for _ in range(d)] for _ in range(c)] for _ in range(b)] for _ in range(a)]



class MemoryStats:
    """用于记录内存使用的统计板"""
    def __init__(self, name: str):
        self.name = name
        self.current_bytes = 0
        self.peak_bytes = 0

    def update(self, delta_bytes: int):
        # 通过传入负值，可以记录当前内存使用量的减少
        self.current_bytes += delta_bytes
        if self.current_bytes > self.peak_bytes:
            self.peak_bytes = self.current_bytes

    def reset(self):
        self.current_bytes = 0
        self.peak_bytes = 0


class MonitoredValueHolder:
    """
    ValueHolder 的装饰器/代理类。
    在 store/pop 数据时自动向 MemoryStats 汇报内存变化。
    """
    def __init__(self, stats: MemoryStats):
        self.holder = ValueHolder()
        self.stats = stats
        self._stored_bytes = 0

    @property
    def val(self):
        return self.holder.val
    
    @val.setter
    def val(self, new_value):
        # 写操作：拦截赋值，强制走 store 逻辑进行计费
        if self._stored_bytes > 0:
            self.clear()
        self.store(new_value)

    def _get_size(self, val):
        """计算张量或数据的字节大小"""
        if val is None:
            return 0
        
        # 递归处理 list 和 tuple
        if isinstance(val, (list, tuple)):
            return sum(self._get_size(v) for v in val)
        
        # 处理 PyTorch Tensor
        if hasattr(val, 'element_size') and hasattr(val, 'numel'):
            return val.element_size() * val.numel()
        
        # 处理 Numpy Array
        if hasattr(val, 'nbytes'):
            return val.nbytes
            
        # 处理自定义 Tensor 包装类 (假设有 bytes 属性或 shape/dtype)
        # 这里需要根据 flexllmgen 实际的 Tensor 类进行适配
        if hasattr(val, 'bytes'): 
            return val.bytes
        if hasattr(val, 'shape') and hasattr(val, 'dtype'):
            # 简单的兜底估算
            try:
                dtype_size = val.dtype.itemsize
                return np.prod(val.shape) * dtype_size
            except:
                pass
                
        # 如果无法计算，为了不报错，打印警告并返回0
        # print(f"Warning: Could not calculate size for {type(val)}")
        return 0

    def store(self, val):
        self._stored_bytes = self._get_size(val)
        self.stats.update(self._stored_bytes)
        self.holder.store(val)

    def pop(self):
        if self._stored_bytes > 0:
            self.stats.update(-self._stored_bytes)
            self._stored_bytes = 0
        val = self.holder.pop()
        return val

    def clear(self):
        if self._stored_bytes > 0:
            self.stats.update(-self._stored_bytes)
            self._stored_bytes = 0
        self.holder.clear()

    # 代理所有未覆盖的属性访问 (如 .val) 到原始 holder
    def __getattr__(self, name):
        return getattr(self.holder, name)
    
    def __setattr__(self, name, value):
        # 避免无限递归，处理自身属性
        if name in ['holder', 'stats', '_stored_bytes']:
            super().__setattr__(name, value)
        elif name == 'val': 
            # 如果直接设置 .val (如 update_attention_mask 中)，也需要尝试追踪
            # 但通常不建议直接设置 .val，这里做个简单代理
            self.holder.val = value
            # 注意：直接设置 val 很难追踪 size 变化，建议尽量用 store
        else:
            setattr(self.holder, name, value)


class MonitoredBuffer:
    """
    能够记录内存使用变化的缓冲区容器，用于替代array_2d等函数创建的多维列表
    """
    def __init__(self, shape: tuple, name: str):
        self.name = name
        self.stats = MemoryStats(name)
        self.shape = shape
        self.ndim = len(shape)
        
        # 递归构建多维列表，且所有元素共享同一个 self.stats
        self._data = self._build_recursive(shape, self.stats)

    def _build_recursive(self, shape, stats):
        if len(shape) == 1:
            # 最后一维，创建实际的 MonitoredValueHolder
            return [MonitoredValueHolder(stats) for _ in range(shape[0])]
        else:
            # 递归创建下一维
            return [self._build_recursive(shape[1:], stats) for _ in range(shape[0])]

    def __getitem__(self, index):
        """支持索引操作 self.cache_home[i]"""
        return self._data[index]

    def __len__(self):
        return self.shape[0]

    # --- 将 MemoryStats 的方法暴露给容器本身 ---
    
    @property
    def peak_bytes(self):
        return self.stats.peak_bytes
    
    @property
    def current_bytes(self):
        return self.stats.current_bytes

    def reset_stats(self):
        self.stats.reset()

    def __repr__(self):
        return str(self.stats)



def vector_gather(vectors, indices):
    """
    Gathers (batched) vectors according to indices.
    Arguments:
        vectors: Tensor[S, B, H]
        indices: Tensor[K, B]
    Returns:
        Tensor[K, B, H]
    """
    S, B, H = vectors.shape
    K, B2 = indices.shape
    assert B == B2
    indices = indices.reshape(K, B, 1).expand(K, B, H)
    out = vectors.gather(dim=0, index=indices)
    return out


def run_cmd(cmd):
    print(cmd)
    os.system(cmd)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def project_decode_latency(costs, prompt_len, gen_len):
    decode_costs = costs[1:]

    if gen_len / prompt_len < 0.1:
        warmup = 2
        decode_latency = (sum(decode_costs[:warmup]) +
            np.mean(decode_costs[warmup:]) * (gen_len - 1 - warmup))
    else:
        warmup = 2
        decode_latency = (sum(decode_costs[:warmup]) +
            np.mean(decode_costs[warmup:]) * (gen_len - 1 - warmup))

        #assert len(decode_costs) >= 4
        #warmup = 2
        #xs = np.arange(warmup, len(decode_costs))
        #ys = np.asarray(decode_costs[warmup:])
        #curve = np.poly1d(np.polyfit(xs, ys, deg=1))
        #ys_pred = [curve(x) for x in range(gen_len-1)]
        #decode_latency = sum(ys_pred)

        #print([round(x, 4) for x in decode_costs])
        #print([round(x, 4) for x in ys_pred])

    return decode_latency


def write_benchmark_log(filename, model_size, cache_size, hidden_size,
        gpu_peak_mem, projected, prefill_latency, prefill_throughput,
        decode_latency, decode_throughput, total_latency, total_throughput):

    log_str = (f"model size: {model_size/GB:.3f} GB\t"
               f"cache size: {cache_size/GB:.3f} GB\t"
               f"hidden size (p): {hidden_size/GB:.3f} GB\n"
               f"peak gpu mem: {gpu_peak_mem / GB:.3f} GB\t"
               f"projected: {projected}\n"
               f"prefill latency: {prefill_latency:.3f} s\t"
               f"prefill throughput: {prefill_throughput:.3f} token/s\n"
               f"decode latency: {decode_latency:.3f} s\t"
               f"decode throughput: {decode_throughput:.3f} token/s\n"
               f"total latency: {total_latency:.3f} s\t"
               f"total throughput: {total_throughput:.3f} token/s")
    with open(filename, "a") as fout:
        fout.write(log_str + "\n")

    return log_str


def read_benchmark_log(filename):
    with open(filename) as fin:
        lines = fin.readlines()

    def extract(line):
        a, b = line.split("\t")
        latency = a[a.index(":") + 1:a.index(" s")]
        throughput = b[b.index(":") + 1:b.index(" to")]
        return float(latency), float(throughput)

    prefill_latency, prefill_throughput = extract(lines[2])
    decode_latency, decode_throughput = extract(lines[3])
    total_latency, total_throughput = extract(lines[4])

    return BenchmarkResult(
        prefill_latency, prefill_throughput,
        decode_latency, decode_throughput,
        total_latency, total_throughput,
    )



class CaptureAttnWeight:
    '''
    Usage:
    SAVE: init -> setup -> add -> save
    LOAD: init -> setup -> load

    Note:
    layer range from 0, step range from 1.
    attn_weight shape: (n_head, seq)
    '''
    save_path: str
    data: dict
    max_step: int

    def __init__(self):
        self.save_path = None
        self.data = {}
        self.max_step = 0

    def setup(self, model_type, video_id):
        save_dir = '/data1/lyc/flexllmgen_outputs/attn_weight'
        self.save_path = f'{save_dir}/{model_type}_{video_id}/{model_type}_{video_id}.pt'

    def add(self, attn_weight, layer, step):
        self.data[f'layer{layer}_step{step}'] = attn_weight
        if step > self.max_step:
            self.max_step = step

    def clear(self):
        self.save_path = None
        self.data = {}
        self.max_step = 0

    def save(self):
        torch.save(self.data, self.save_path)

    def load(self):
        self.data = torch.load(self.save_path)

    def get(self, layer, step):
        return self.data.get(f'layer{layer}_step{step}')
    
total_attn_weight = CaptureAttnWeight()