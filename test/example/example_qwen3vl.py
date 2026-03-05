from dotenv import load_dotenv
load_dotenv(override=True)

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


device = "cuda:0"
model_path = "/data/lyc/models/Qwen3-VL-8B-Instruct"

processor = AutoProcessor.from_pretrained(model_path)


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
                # "max_pixels": 360 * 420,
                # "fps": 1.0,
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
inputs = inputs.to(device)

print(f'{inputs.input_ids.shape=}')
# print(f'{inputs.input_ids.dtype=}')
# print(f'{inputs.attention_mask.shape=}')
# print(f'{inputs.attention_mask.dtype=}')
print(f'{inputs.pixel_values_videos.shape=}')
# print(f'{inputs.pixel_values_videos.dtype=}')
# print(f'{inputs.video_grid_thw.shape=}')
# print(f'{inputs.video_grid_thw.dtype=}')
print(f'{inputs.video_grid_thw=}')

# Load Model
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