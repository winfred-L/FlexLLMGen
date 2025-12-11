import av
import torch
import numpy as np
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration


model_path = '/data/lyc/models/LLaVA-NeXT-Video-7B-hf'
device = 'cuda:0'

model = LlavaNextVideoForConditionalGeneration.from_pretrained(
    model_path,
    attn_implementation = 'eager',
    dtype = torch.float16, 
    low_cpu_mem_usage = True, 
).to(device)
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


# video_path = '/data/lyc/datasets/Video-MME/video/ZHWZf1Z4B5k.mp4' #28s
# video_path = "/data/lyc/datasets/Video-MME/video/zNxi2s36tS0.mp4" #43s
# video_path = "/data/lyc/datasets/Video-MME/video/Z-rHofd6g2Q.mp4" #66s
video_path = "/data1/lyc/datasets/mlvu_test/MLVU_Test/video/test_game_1.mp4" #5min16s
# video_path = "/data1/lyc/datasets/mlvu_test/MLVU_Test/video/test_food_3.mp4" #6min51s
# video_path = "/data1/lyc/datasets/mlvu_test/MLVU_Test/video/test_AWB-6.mp4" #7min30s
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
target_fps = 0.25 #1
step = src_fps / target_fps
if step < 1:
    step = 1
indices = np.arange(0, total_frames, step).astype(int)
clip = read_video_pyav(container, indices)
inputs = processor(text=prompt, videos=clip, padding=True, return_tensors="pt").to(model.device)
'''
inputs is a dict of
    input_ids: tensor[batch_size, seq_len]
    attention_mask: tensor[batch_size, seq_len]
    pixel_values_videos: tensor[batch_size, num_frames, 3, H, W]
'''

print(inputs.input_ids.shape)

with torch.inference_mode():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=False,
    )
output_text = processor.decode(output_ids[0], skip_special_tokens=True)
output_text = output_text.split("ASSISTANT: ")[1]

print(output_text)