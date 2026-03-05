from dotenv import load_dotenv
load_dotenv(override=True)

import argparse
import sys
from datetime import datetime
from typing import List, Tuple
from tqdm import tqdm

import torch
from transformers import AutoProcessor
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    print("qwen is not installed. Please install qwen-vl-utils to use this model.")

# FlexGen imports
from flexllmgen.pytorch_backend import (TorchDevice, TorchDisk, TorchMixedDevice)
from flexllmgen.utils import ExecutionEnv
from flexllmgen.policy import Policy
from flexllmgen.models.qwen25vl import Qwen25VLFlexLM
from flexllmgen.models.qwen25vl_config import get_qwen25vl_config
from flexllmgen.models.qwen3vl import Qwen3VLFlexLM
from flexllmgen.models.qwen3vl_config import get_qwen3vl_config
from flexllmgen.main import add_parser_arguments



device = "cuda:0"
model_path = "/data/lyc/models/Qwen3-VL-8B-Instruct"


video_path = "/data1/lyc/hf_home/lvbench/Cm73ma6Ibcs.mp4" # 1 hour
question = 'Please describe this video in detail.'


# set up argument parser
parser = argparse.ArgumentParser()
add_parser_arguments(parser)
args = parser.parse_args([])

# initialize processor
processor = AutoProcessor.from_pretrained(model_path)

# prepare execution environment
gpu = TorchDevice(args.cuda_device)
cpu = TorchDevice("cpu")
disk = TorchDisk(args.offload_dir)
env = ExecutionEnv(gpu=gpu, cpu=cpu, disk=disk, mixed=TorchMixedDevice([gpu, cpu, disk]))
policy = Policy(args.gpu_batch_size, args.num_gpu_batches,
                        args.percent[0], args.percent[1],
                        args.percent[2], args.percent[3],
                        args.percent[4], args.percent[5],
                        args.overlap, args.sep_layer, args.pin_weight,
                        args.cpu_cache_compute, args.attn_sparsity,
                        args.compress_weight, None,
                        args.compress_cache, None,
                        args.attn_impl,
                        args.do_sparse, args.threshold_S, args.threshold_D)

# initialize model
model = Qwen3VLFlexLM(args.model_type, env, args.path, policy)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": video_path,
                # restrict the resolution of individual frames in the video
                "min_pixels": 4 * 32 * 32,
                "max_pixels": 100 * 32 * 32,
                # limit the total number of tokens in the video
                "total_pixels": 100 * 1024 * 32 * 32, # 100K tokens
                # accept either `fps` or `nframes`
                # "fps": 2.0,
                "nframes": 2048,
            },
            {"type": "text", "text": question},
        ],
    }
]


text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
images, videos, video_kwargs = process_vision_info(messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)

# each video returns as (video_tensor, video_metadata)
# split the videos and according metadatas
if videos is not None:
    videos, video_metadatas = zip(*videos)
    videos, video_metadatas = list(videos), list(video_metadatas)
else:
    video_metadatas = None


inputs = processor(
    text=text,
    images=images,
    videos=videos,
    video_metadata=video_metadatas,
    return_tensors="pt",
    do_resize=False, # avoid duplicate resizing
    **video_kwargs
)

print(f'{inputs.input_ids.shape=}')


with torch.inference_mode():
    output_ids = model.generate(
        inputs,
        max_new_tokens=args.gen_len,
        do_sample=args.do_sample,
        temperature=args.temperature,
        stop=None,
        debug_mode=args.debug_mode,
        cut_gen_len=args.cut_gen_len)




generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]
print('='*30)
print(output_text)
print('='*30)


peak_mem = torch.cuda.max_memory_allocated(device)
print(f"cuda peak mem: {peak_mem / 1024 / 1024 / 1024 :.4f} GB")