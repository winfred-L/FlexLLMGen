import decord
import torch
from decord import VideoReader, gpu
from torch.utils.dlpack import from_dlpack

def test_gpu_decode(video_path):
    try:
        # 1. 尝试初始化 GPU 上下文
        ctx = gpu(0)
        
        # 2. 加载视频并指定硬件加速宽度/高度（可选）
        vr = VideoReader(video_path, ctx=ctx)
        print(f"✅ Video loaded. Total frames: {len(vr)}")
        
        # 3. 提取 1 帧测试
        frame = vr.get_batch([0])
        
        # 4. 零拷贝转换为 PyTorch Tensor
        frame_torch = from_dlpack(frame.to_dlpack())
        
        print(f"✅ GPU Decoding worked!")
        print(f"Frame shape: {frame_torch.shape}")
        print(f"Frame device: {frame_torch.device}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

# 替换为你环境下的任意视频路径
test_gpu_decode("/data1/lyc/hf_home/lvbench/Cm73ma6Ibcs.mp4")