import torch
import os
import cv2
import time
from pathlib import Path
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

def sample_video_frames(video_path, output_dir, num_frames=1024, target_size=(420, 360)):
    """
    采样视频帧并保存到指定文件夹
    
    Args:
        video_path: 输入视频路径
        output_dir: 输出帧的文件夹路径
        num_frames: 采样帧数
        target_size: 目标分辨率 (height, width)
    
    Returns:
        frame_paths: 保存的帧文件路径列表
    """
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")
    
    # 获取视频总帧数和FPS
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"视频总帧数: {total_frames}, FPS: {fps}")
    
    # 计算采样间隔
    if total_frames <= num_frames:
        # 如果总帧数小于目标帧数，则不跳过帧
        frame_indices = list(range(total_frames))
        sample_fps = fps
    else:
        # 均匀采样
        frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
        # 计算采样后的等效FPS
        sample_fps = fps * num_frames / total_frames
    
    print(f"采样帧数: {len(frame_indices)}, 采样后等效FPS: {sample_fps:.2f}")
    
    # 读取并保存帧
    frame_paths = []
    for idx, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            print(f"警告: 无法读取第 {frame_idx} 帧")
            continue
        
        # 调整分辨率
        frame_resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
        
        # 保存帧
        frame_path = os.path.join(output_dir, f"frame_{idx:05d}.jpg")
        cv2.imwrite(frame_path, frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
        frame_paths.append(frame_path)
    
    cap.release()
    print(f"已保存 {len(frame_paths)} 帧到: {output_dir}")
    
    return frame_paths, sample_fps


# ==================== 主程序 ====================

device = "cuda:0"
model_path = "/data/lyc/models/Qwen3-VL-8B-Instruct"

video_id = "Cm73ma6Ibcs"
video_path = f"/data1/lyc/hf_home/lvbench/{video_id}.mp4"
output_frame_dir = f"/data1/lyc/hf_home/lvbench_images/{video_id}"
if not os.path.exists(output_frame_dir):
    os.makedirs(output_frame_dir)


frame_paths, sample_fps = sample_video_frames(
    video_path=video_path,
    output_dir=output_frame_dir,
    num_frames=2048, #1024,
    target_size=(420, 360)  # (height, width)
)
frame_uris = [f"{os.path.abspath(path)}" for path in frame_paths]


processor = AutoProcessor.from_pretrained(model_path)


question = 'Please describe this video in detail.'
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": frame_uris,
                "sample_fps": sample_fps,
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
inputs = inputs.to(device)

torch.cuda.synchronize()
t2 = time.time()
print(f"t2-t1={t2-t1}")

print(f'{inputs.input_ids.shape=}')
if hasattr(inputs, 'pixel_values_videos'):
    print(f'{inputs.pixel_values_videos.shape=}')
if hasattr(inputs, 'video_grid_thw'):
    print(f'{inputs.video_grid_thw=}')


model = AutoModelForImageTextToText.from_pretrained(
    model_path, dtype=torch.bfloat16, attn_implementation="flash_attention_2",
    device_map=device, low_cpu_mem_usage=True, # accelerate model loading
).eval()

torch.cuda.synchronize()
t3 = time.time()
print(f"t3-t2={t3-t2}")


with torch.inference_mode():
    generated_ids = model.generate(
        **inputs, 
        max_new_tokens=128, 
        do_sample=False
    )

torch.cuda.synchronize()
t4 = time.time()
print(f"t4-t3={t4-t3}")

generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]

output_text = processor.batch_decode(
    generated_ids_trimmed, 
    skip_special_tokens=True, 
    clean_up_tokenization_spaces=False
)[0]
print(output_text)