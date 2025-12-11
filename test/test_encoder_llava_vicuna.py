import av
import torch
import numpy as np
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration


model_path = '/data/lyc/models/LLaVA-NeXT-Video-7B-hf'
device = 'cuda:0'


from transformers import AutoConfig
config = AutoConfig.from_pretrained(
    model_path, 
    trust_remote_code=True
)
# import pdb; pdb.set_trace()
'''
(Pdb) p config
LlavaNextVideoConfig {
  "architectures": [
    "LlavaNextVideoForConditionalGeneration"
  ],
  "dtype": "bfloat16",
  "ignore_index": -100,
  "image_grid_pinpoints": [
    [
      336,
      672
    ],
    [
      672,
      336
    ],
    [
      672,
      672
    ],
    [
      1008,
      336
    ],
    [
      336,
      1008
    ]
  ],
  "image_seq_length": 576,
  "image_token_index": 32001,
  "model_type": "llava_next_video",
  "multimodal_projector_bias": true,
  "projector_hidden_act": "gelu",
  "spatial_pool_mode": "average",
  "spatial_pool_out_channels": 1024,
  "spatial_pool_stride": 2,
  "text_config": {
    "_name_or_path": "lmsys/vicuna-7b-v1.5",
    "architectures": [
      "LlamaForCausalLM"
    ],
    "attention_bias": false,
    "attention_dropout": 0.0,
    "dtype": "float16",
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 4096,
    "initializer_range": 0.02,
    "intermediate_size": 11008,
    "max_position_embeddings": 4096,
    "mlp_bias": false,
    "model_type": "llama",
    "num_attention_heads": 32,
    "num_hidden_layers": 32,
    "num_key_value_heads": 32,
    "pad_token_id": 0,
    "pretraining_tp": 1,
    "rms_norm_eps": 1e-05,
    "rope_scaling": {
      "factor": 2.5,
      "rope_type": "linear",
      "type": "linear"
    },
    "rope_theta": 10000.0,
    "type": "linear",
    "use_cache": true,
    "vocab_size": 32064
  },
  "tie_word_embeddings": false,
  "transformers_version": "4.57.3",
  "use_image_newline_parameter": true,
  "video_seq_length": 288,
  "video_token_index": 32000,
  "vision_config": {
    "attention_dropout": 0.0,
    "hidden_act": "quick_gelu",
    "hidden_size": 1024,
    "image_size": 336,
    "initializer_factor": 1.0,
    "initializer_range": 0.02,
    "intermediate_size": 4096,
    "layer_norm_eps": 1e-05,
    "model_type": "clip_vision_model",
    "num_attention_heads": 16,
    "num_channels": 3,
    "num_hidden_layers": 24,
    "patch_size": 14,
    "projection_dim": 768,
    "vocab_size": 32000
  },
  "vision_feature_layer": -2,
  "vision_feature_select_strategy": "default"
}
'''





# model = LlavaNextVideoForConditionalGeneration.from_pretrained(
#     model_path,
#     attn_implementation = 'eager',
#     dtype = torch.float16, 
#     low_cpu_mem_usage = True, 
# ).to(device)
processor = LlavaNextVideoProcessor.from_pretrained(model_path, use_fast=True)



def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


video_path = '/data/lyc/datasets/Video-MME/video/ZHWZf1Z4B5k.mp4'
question = 'Please describe this video in detail.'
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": question},
            {"type": "video"},
        ],
    },
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
container = av.open(video_path)

# sample uniformly x frames from the video, can sample more for longer videos
video_stream = container.streams.video[0]
total_frames = video_stream.frames
src_fps = float(video_stream.average_rate)
target_fps = 1
step = src_fps / target_fps
if step < 1:
    step = 1
indices = np.arange(0, total_frames, step).astype(int)
clip = read_video_pyav(container, indices)
inputs = processor(text=prompt, videos=clip, padding=True, return_tensors="pt")
'''
inputs is a dict of
    input_ids: tensor[batch_size, seq_len]
    attention_mask: tensor[batch_size, seq_len]
    pixel_values_videos: tensor[batch_size, num_frames, 3, H, W]
'''

import pdb; pdb.set_trace()