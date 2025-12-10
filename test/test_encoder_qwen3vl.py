import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

model_path = "/data/lyc/models/Qwen3-VL-8B-Instruct"
device = "cuda:1"

from transformers import AutoConfig
config = AutoConfig.from_pretrained(
    model_path, 
    trust_remote_code=True
)
# import pdb; pdb.set_trace()
'''
(Pdb) p config
Qwen3VLConfig {
  "architectures": [
    "Qwen3VLForConditionalGeneration"
  ],
  "image_token_id": 151655,
  "model_type": "qwen3_vl",
  "text_config": {
    "attention_bias": false,
    "attention_dropout": 0.0,
    "bos_token_id": 151643,
    "dtype": "bfloat16",
    "eos_token_id": 151645,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 4096,
    "initializer_range": 0.02,
    "intermediate_size": 12288,
    "max_position_embeddings": 262144,
    "model_type": "qwen3_vl_text",
    "num_attention_heads": 32,
    "num_hidden_layers": 36,
    "num_key_value_heads": 8,
    "rms_norm_eps": 1e-06,
    "rope_scaling": {
      "mrope_interleaved": true,
      "mrope_section": [
        24,
        20,
        20
      ],
      "rope_type": "default"
    },
    "rope_theta": 5000000,
    "use_cache": true,
    "vocab_size": 151936
  },
  "tie_word_embeddings": false,
  "transformers_version": "4.57.3",
  "video_token_id": 151656,
  "vision_config": {
    "deepstack_visual_indexes": [
      8,
      16,
      24
    ],
    "depth": 27,
    "hidden_act": "gelu_pytorch_tanh",
    "hidden_size": 1152,
    "in_channels": 3,
    "initializer_range": 0.02,
    "intermediate_size": 4304,
    "model_type": "qwen3_vl",
    "num_heads": 16,
    "num_position_embeddings": 2304,
    "out_hidden_size": 4096,
    "patch_size": 16,
    "spatial_merge_size": 2,
    "temporal_patch_size": 2
  },
  "vision_end_token_id": 151653,
  "vision_start_token_id": 151652
}
'''


processor = AutoProcessor.from_pretrained("/data/lyc/models/Qwen3-VL-8B-Instruct")


video_fps = 1.0
video_path = '/data/lyc/datasets/Video-MME/video/ZHWZf1Z4B5k.mp4'
question = 'Please describe this video in detail.'
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": video_path,
                "max_pixels": 360 * 420,
                "fps": video_fps,
            },
            {"type": "text", "text": question},
        ],
    }
]

# Preparation for inference
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
import pdb; pdb.set_trace()
'''
(Pdb) p type(inputs)
<class 'transformers.feature_extraction_utils.BatchFeature'>
input_ids: tensor, torch.Size([1, 11597])
attention_mask: tensor, torch.Size([1, 11597])
pixel_values_videos: tensor, torch.Size([45360, 1536])
    # 45360(patch count) = 28*30*54 (video_grid_thw)
    # 1536(patch size) = 2(time) * 16(height) * 16(width) * 3(channels)
    # "temporal_patch_size": 2
    # "patch_size": 16
    # "in_chans": 3
video_grid_thw: tensor, torch.Size([1, 3])

(Pdb) p inputs.keys()
KeysView({
    'input_ids': tensor([[151644,    872,    198,  ..., 151644,  77091,    198]]),
    'attention_mask': tensor([[1, 1, 1,  ..., 1, 1, 1]]),
    'pixel_values_videos': tensor([[ 0.2000,  0.2000,  0.2078,  ..., -0.5529, -0.5529, -0.5529],
        [ 0.3961,  0.4118,  0.4275,  ..., -0.5529, -0.5529, -0.5451],
        [ 0.1216,  0.1294,  0.1373,  ..., -0.3725, -0.3725, -0.3725],
        ...,
        [-0.1686, -0.1765, -0.1922,  ..., -1.0000, -1.0000, -1.0000],
        [-0.1216, -0.1216, -0.1294,  ..., -1.0000, -1.0000, -1.0000],
        [-0.1608, -0.1608, -0.1608,  ..., -1.0000, -1.0000, -1.0000]]),
    'video_grid_thw': tensor([[28, 30, 54]])})

'''



from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel


















# # Inference: Generation of the output
# generated_ids = model.generate(**inputs, max_new_tokens=128)
# generated_ids_trimmed = [
#     out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
# ]
# output_text = processor.batch_decode(
#     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
# )
# print(output_text)