import torch
import flexllmgen.utils as utils

model_type = 'qwen3vl-8b'
video_id = '28s'
utils.total_attn_weight.setup(model_type, video_id)
utils.total_attn_weight.load()

if model_type == 'qwen3vl-8b':
    num_layer = 36
    num_head = 32
    num_kv_head = 8


# print(utils.total_attn_weight.data)
print(utils.total_attn_weight.get(layer=1, step=1).shape)

import pdb; pdb.set_trace()