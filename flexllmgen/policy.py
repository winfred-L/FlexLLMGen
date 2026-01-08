
import numpy as np
import dataclasses

from flexllmgen.compression import CompressionConfig
from flexllmgen.utils import torch_dtype_to_np_dtype




DUMMY_WEIGHT = "_DUMMY_"  # Use dummy weights for benchmark purposes
'''
DUMMY_WEIGHT 可以实现，没有下载实际模型权重文件的情况下测试系统的性能。
- 在初始化权重列表时，会跳过文件读取，直接用全 1 的数据（np.ones）填充权重张量，以模拟真实负载、避免磁盘 I/O 开销、缺少文件报错、防止自动下载权重
- 在推理结束后的结果处理阶段，不会使用 Tokenizer 对输出的乱码进行解码
'''



@dataclasses.dataclass(frozen=True)
class Policy:
    '''
    用于存储推理过程中的所有策略配置
    '''
    gpu_batch_size: int
    num_gpu_batches: int

    # percent = a means a%
    # 定义了权重（w）、KV Cache（cache）和激活值（act）在 GPU、CPU 和磁盘上的分布百分比
    w_gpu_percent: float
    w_cpu_percent: float
    cache_gpu_percent: float
    cache_cpu_percent: float
    act_gpu_percent: float # must be 100 or 0
    act_cpu_percent: float # must be 100 or 0

    # Whether to overlap the I/O and compute
    overlap: bool

    # Whether to separate attention and mlp as two layers
    sep_layer: bool

    # Whether to use pinned memory for weights on CPU
    pin_weight: bool

    # Whether to compute attention on CPU
    cpu_cache_compute: bool

    # Sparsity of attention weights (Unused)
    attn_sparsity: float

    # Compress weights with group-wise quantization
    compress_weight: bool
    comp_weight_config: CompressionConfig

    # Compress KV cache with group-wise quantization
    compress_cache: bool
    comp_cache_config: CompressionConfig

    # Attention implementation
    attn_impl: str

    # Sparsity settings
    do_sparse: bool
    threshold_S: float
    threshold_D: float

    @property
    def w_disk_percent(self):
        return 100 - self.w_gpu_percent - self.w_cpu_percent

    @property
    def cache_disk_percent(self):
        return 100 - self.cache_gpu_percent - self.cache_cpu_percent

    @property
    def act_disk_percent(self):
        return 100 - self.act_gpu_percent - self.act_cpu_percent


def get_choice(cur_percent, percents, choices):
    '''
    根据当前的百分比进度，决定某个具体的 Tensor 应该存放在哪里
    '''
    percents = np.cumsum(percents) # 计算 percents（如 [磁盘%, CPU%, GPU%]）的累积和（cumsum）
    assert np.abs(percents[-1] - 100) < 1e-5

    # 遍历累积和数组，如果 cur_percent 小于当前的累积值，就返回对应的 choice（即对应的设备对象 env.disk, env.cpu 或 env.gpu）
    for i in range(len(percents)):
        if cur_percent < percents[i]:
            return choices[i]
    return choices[-1]


def init_weight_list(weight_specs, policy, env):
    '''
    初始化并分配一组权重的内存。这是实现“按比例卸载权重”的核心函数
    '''
    # 计算所有权重的总大小，并计算累积大小
    dev_percents = [policy.w_disk_percent, policy.w_cpu_percent, policy.w_gpu_percent]
    dev_choices = [env.disk, env.cpu, env.gpu]

    sizes = [np.prod(spec[0]) for spec in weight_specs]
    sizes_cumsum = np.cumsum(sizes)

    # 遍历每个权重规格 weight_specs
    ret = []
    for i in range(len(weight_specs)):
        mid_percent = (sizes_cumsum[i] - sizes[i] / 2) / sizes_cumsum[-1]
        home = get_choice(mid_percent * 100, dev_percents, dev_choices)
        shape, dtype, filename = weight_specs[i]

        if len(shape) < 2:
            pin_memory = True
            compress = False
        else:
            pin_memory = policy.pin_weight
            compress = policy.compress_weight

        if not compress:
            weight = home.allocate(shape, dtype, pin_memory=pin_memory)

            if DUMMY_WEIGHT not in filename:
                weight.load_from_np_file(weight_specs[i][2])
            else:
                weight.load_from_np(np.ones(shape, dtype))
                #weight.load_from_np(np.random.rand(*shape).astype(dtype))
        else:
            weight = home.compressed_device.allocate(
                shape, dtype, policy.comp_weight_config, pin_memory=pin_memory)

            if DUMMY_WEIGHT not in filename:
                weight.load_from_np_file(weight_specs[i][2])
            else:
                for i in range(2):
                    x = weight.data[i]
                    x.load_from_np(np.ones(x.shape, torch_dtype_to_np_dtype[x.dtype]))

        ret.append(weight)
    return ret
