from dotenv import load_dotenv
load_dotenv(override=True)

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

device = "cuda:0"
# model_path = "Qwen/Qwen3-VL-8B-Instruct"
model_path = "/data/lyc/models/Qwen3-VL-8B-Instruct"


max_pixels: int = 224 * 1024 * 32 * 32 # 224K tokens
min_pixels: int = 224 * 1024 * 32 * 32
processor = AutoProcessor.from_pretrained(
    model_path,
    max_pixels=max_pixels,
    min_pixels=min_pixels,
    use_fast=True,
)



# video_path = "./test/video/28s.mp4"
video_path = "/data1/lyc/hf_home/lvbench/Cm73ma6Ibcs.mp4"
question = 'Please describe this video in detail.'
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": video_path,
                "max_pixels": 640 * 2 * 28 * 28, # 一帧最多 640 token
                "fps": 2.0,
                "nframes": 2048, # 最多 2048 帧
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
    return_tensors="pt",
    # max_pixels=1280 * 28 * 28,      # 每帧最大像素（≈1280 tokens）
    # fps=2.0,                        # 采样帧率
    # max_frames=2048, 
)
print(f'{inputs.input_ids.shape=}')
print(f'{inputs.pixel_values_videos.shape=}')
print(f'{inputs.video_grid_thw=}')
import pdb; pdb.set_trace()

inputs = inputs.to(device)


model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
).to(device).eval()

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]
print(output_text)