from dotenv import load_dotenv
load_dotenv(override=True)

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    print("qwen is not installed. Please install qwen-vl-utils to use this model.")

import torch
from decord import VideoReader, gpu
from PIL import Image
import numpy as np

def get_video_inputs_with_metadata(video_path, nframes):
    vr = VideoReader(video_path, ctx=gpu(0))
    total_frames = len(vr)
    avg_fps = vr.get_avg_fps()
    duration = total_frames / avg_fps 
    
    indices = np.linspace(0, total_frames - 1, nframes, dtype=int)
    video_ndarray = vr.get_batch(indices) 
    video_tensor = torch.from_numpy(video_ndarray.asnumpy()).permute(0, 3, 1, 2)
    
    logical_fps = nframes / duration
    
    metadata = {
        "fps": logical_fps,
        "total_num_frames": nframes,
    }
    
    return video_tensor, metadata





import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

device = "cuda:0"


# processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
# max_pixels: int = 16384 * 28 * 28 * 16
max_pixels: int = 224 * 1024 * 32 * 32
# max_pixels: int = 224 * 1024 * 2 * 28 * 28 # 224K tokens
# min_pixels: int = 32*28*28
min_pixels: int = 224 * 1024 * 32 * 32
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct",
    max_pixels=max_pixels,
    min_pixels=min_pixels,
    use_fast=True
)


video_path = '/data1/lyc/hf_home/lvbench/Cm73ma6Ibcs.mp4'
question = 'Please describe this video in detail.'




video_inputs, video_metadata = get_video_inputs_with_metadata(video_path, nframes=2048)
print(f"{video_inputs.shape=}")
print(f"{video_metadata=}")


messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": video_inputs, # 这里的 video 已经是 tensor 了
                "fps": video_metadata["fps"], # 传入 fps
            },
            {"type": "text", "text": "describe this video in detail."},
        ],
    }
]

text_prompt = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
inputs = processor(
    text=[text_prompt],
    videos=[video_inputs],
    video_metadata=[{
        "fps": video_metadata["fps"], 
        "total_num_frames": video_inputs.shape[0]
    }],
    padding=True,
    return_tensors="pt",
)



print(f'{inputs.input_ids.shape=}')
print(f'{inputs.pixel_values_videos.shape=}')
print(f'{inputs.video_grid_thw=}')
import pdb; pdb.set_trace()


inputs = inputs.to(device)




model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct",
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