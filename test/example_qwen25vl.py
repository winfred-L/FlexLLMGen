import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    print("qwen is not installed. Please install qwen-vl-utils to use this model.")

model_path = '/data/lyc/models/Qwen2.5-VL-7B-Instruct'
device = "cuda:0"
max_pixels: int = 16384*28*28
min_pixels: int = 32*28*28

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    attn_implementation="eager",
    trust_remote_code=True,
).to(device).eval()
processor = AutoProcessor.from_pretrained(model_path, max_pixels=max_pixels, min_pixels=min_pixels, use_fast=True)


# prepare input
# video_path = '/data/lyc/datasets/Video-MME/video/ZHWZf1Z4B5k.mp4' #28s
# video_path = "/data/lyc/datasets/Video-MME/video/zNxi2s36tS0.mp4" #43s
video_path = "/data/lyc/datasets/Video-MME/video/Z-rHofd6g2Q.mp4" #66s
question = 'Please describe this video in detail.'
video_fps = 1.0
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

text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
    **video_kwargs,
)
inputs = inputs.to(model.device)

# inference
with torch.inference_mode():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=False,
    )

# process output
output_ids = [output_ids[0, len(inputs.input_ids[0]) :]]
output_text = processor.batch_decode(
    output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]
print(output_text)