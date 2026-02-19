# Qwen3-VL introduces the new video processor with `video_metadata`
# see https://github.com/QwenLM/Qwen3-VL?tab=readme-ov-file#new-qwen-vl-utils-usage


from dotenv import load_dotenv
load_dotenv(override=True)

import time

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

device = "cuda:0"
model_path = "/data/lyc/models/Qwen3-VL-8B-Instruct"

processor = AutoProcessor.from_pretrained(model_path)


# video_path = "./test/video/28s.mp4"
video_path = "/data1/lyc/hf_home/lvbench/Cm73ma6Ibcs.mp4" # 1 hour
question = 'Please describe this video in detail.'

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": video_path,
                # restrict the resolution of individual frames in the video
                "min_pixels": 4 * 32 * 32,
                "max_pixels": 56 * 32 * 32, #640 * 32 * 32, #256 * 32 * 32,
                # limit the total number of tokens in the video
                "total_pixels": 56 * 1024 * 32 * 32, #224 * 1024 * 32 * 32, # 224K tokens
                # accept either `fps` or `nframes`
                # "fps": 2.0,
                "nframes": 2048,
            },
            {"type": "text", "text": question},
        ],
    }
]

torch.cuda.synchronize()
t0 = time.time()

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
images, videos, video_kwargs = process_vision_info(messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)

# each video returns as (video_tensor, video_metadata)
# split the videos and according metadatas
if videos is not None:
    videos, video_metadatas = zip(*videos)
    videos, video_metadatas = list(videos), list(video_metadatas)
else:
    video_metadatas = None

torch.cuda.synchronize()
t1 = time.time()
print(f"t1-t0={t1-t0}")


inputs = processor(
    text=text,
    images=images,
    videos=videos,
    video_metadata=video_metadatas,
    return_tensors="pt",
    do_resize=False, # avoid duplicate resizing
    **video_kwargs
)

torch.cuda.synchronize()
t2 = time.time()
print(f"t2-t1={t2-t1}")


print(f'{inputs.input_ids.shape=}')
# print(f'{inputs.input_ids.dtype=}')
# print(f'{inputs.attention_mask.shape=}')
# print(f'{inputs.attention_mask.dtype=}')
print(f'{inputs.pixel_values_videos.shape=}')
# print(f'{inputs.pixel_values_videos.dtype=}')
# print(f'{inputs.video_grid_thw.shape=}')
# print(f'{inputs.video_grid_thw.dtype=}')
print(f'{inputs.video_grid_thw=}')
# import pdb; pdb.set_trace()


inputs = inputs.to(device)
# import pdb; pdb.set_trace()


# model = Qwen3VLForConditionalGeneration.from_pretrained(
#     model_path,
#     dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
# ).to(device).eval()

model = AutoModelForImageTextToText.from_pretrained(
    model_path, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
).to(device).eval()

torch.cuda.synchronize()
t3 = time.time()
print(f"t3-t2={t3-t2}")

# Inference: Generation of the output
with torch.inference_mode():
    generated_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)

torch.cuda.synchronize()
t4 = time.time()
print(f"t4-t3={t4-t3}")

generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]
print(output_text)


peak_mem = torch.cuda.max_memory_allocated(device)
print(f"cuda peak mem: {peak_mem / 1024 / 1024 / 1024 :.4f} GB")