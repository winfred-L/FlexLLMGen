from abc import ABC, abstractmethod

class BaseLayer(ABC):
    def __init__(self, config, env, policy):
        self.config = config
        self.env = env
        self.policy = policy
        self.compute = self.env.gpu
        self.weight_load_dst = (self.compute.compressed_device 
            if policy.compress_weight
            else self.compute)
        self.task = None

    def set_task(self, task):
        self.task = task

    @abstractmethod
    def init_weight(self, weight_home, path):
        pass

    @abstractmethod
    def load_weight(self, weight_home, weight_read_buf, k):
        pass

    def store_weight(self, weight_home, weight_write_buf, k):
        pass

    def init_cache_one_gpu_batch(self, cache_home):
        pass

    def load_cache(self, cache_home, cache_read_buf, i):
        pass

    def store_cache(self, cache_home, cache_write_buf, i):
        pass

    @abstractmethod
    def input_act_shape_and_dtype(self, batch_size, seq_len):
        pass

    @abstractmethod
    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k):
        pass