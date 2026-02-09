# import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)
# warnings.simplefilter(action='ignore', category=UserWarning)

import os
MAX_FRAMES = 2048
os.environ["MAX_FRAMES"] = f"{MAX_FRAMES}"

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

# LMMS-Eval imports
from lmms_eval.api.registry import register_model
from lmms_eval.api.model import lmms
from lmms_eval.api.instance import Instance


# import cv2
# from PIL import Image
# import numpy as np

# def get_video_frames(video_path, nframes):
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         return []
    
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     # 计算采样索引：从 0 到 total_frames-1 均匀取 nframes 个点
#     indices = np.linspace(0, total_frames - 1, nframes, dtype=int)
    
#     frames = []
#     for idx in indices:
#         cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
#         ret, frame = cap.read()
#         if not ret:
#             break
#         # OpenCV 是 BGR，需要转为 RGB 供 PIL/Model 使用
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         frames.append(Image.fromarray(frame))
    
#     cap.release()
#     return frames


import time
def print_timing_stats(timings):
    total_time = sum(timings.values())
    print("\n" + "="*50)
    print(f"{'Step':<20} | {'Time (s)':<10} | {'Ratio (%)':<10}")
    print("-" * 50)
    for step, t in timings.items():
        ratio = (t / total_time) * 100
        print(f"{step:<20} | {t:.4f}     | {ratio:.2f}%")
    print("-" * 50)
    print(f"{'Total':<20} | {total_time:.4f}     | 100.00%")
    print("="*50 + "\n")




@register_model("flexgenlm")
class FlexGenLM(lmms):
    def __init__(self, **kwargs):
        super().__init__()

        # set up argument parser
        parser = argparse.ArgumentParser()
        add_parser_arguments(parser)
        args = parser.parse_args([])

        for k, v in kwargs.items():
            if hasattr(args, k):
                target_type = type(getattr(args, k))
                if target_type == bool:
                    if isinstance(v, str):
                        v = v.lower() == 'true'
                elif target_type == list:
                    if isinstance(v, str):
                        # 处理 percent 参数，例如 "100,0,100,0,100,0"
                        v = [int(x) for x in v.split(',')]
                elif target_type == int:
                    v = int(v)
                elif target_type == float:
                    v = float(v)
                setattr(args, k, v)

        assert args.gpu_batch_size == 1
        assert args.num_gpu_batches == 1

        # initialize processor
        if args.model_type == 'qwen25vl-7b':
            model_path = "/data/lyc/models/Qwen2.5-VL-7B-Instruct"
            max_pixels: int = 16384*28*28
            min_pixels: int = 32*28*28
            processor = AutoProcessor.from_pretrained(
                model_path,
                max_pixels=max_pixels,
                min_pixels=min_pixels,
                use_fast=True
            )
            model_config = get_qwen25vl_config(args.model_type)
        elif args.model_type == 'qwen3vl-8b':
            model_path = "/data/lyc/models/Qwen3-VL-8B-Instruct"
            max_pixels: int = 224 * 1024 * 32 * 32 # 224K tokens     #16384*28*28
            min_pixels: int = 32*28*28
            processor = AutoProcessor.from_pretrained(
                model_path,
                max_pixels=max_pixels,
                min_pixels=min_pixels,
                use_fast=True
            )
            model_config = get_qwen3vl_config(args.model_type)

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
        if args.model_type == 'qwen25vl-7b':
            model = Qwen25VLFlexLM(args.model_type, env, args.path, policy)
        elif args.model_type == 'qwen3vl-8b':
            model = Qwen3VLFlexLM(args.model_type, env, args.path, policy)

        # warm up model
        # TODO

        self.args = args
        self.env = env
        self.model = model
        self.processor = processor
        self.default_gen_kwargs = {
            "max_new_tokens": self.args.gen_len,
            "do_sample": self.args.do_sample,
            "temperature": self.args.temperature,
        }


    def generate_until(self, requests: list[Instance]) -> list[str]:
        results = []
        pbar = tqdm(total=len(requests), desc="Model Responding")
        try:
            for request in requests:
                timings = {
                    "Data Prep": 0.0, 
                    "Preprocessing": 0.0,
                    "Inference": 0.0,
                    "Decoding": 0.0
                }
                torch.cuda.synchronize() # 确保之前任务完成
                t0 = time.time()

                # parse request
                res_args = request.args
                context = res_args[0] # text input (question)
                gen_kwargs = res_args[1]
                doc_to_visual = res_args[2]
                doc_id = res_args[3]
                task = res_args[4]
                split = res_args[5]

                # prepare inputs
                doc = self.task_dict[task][split][doc_id]
                video_path = doc_to_visual(doc)
                video_path = video_path[0] if isinstance(video_path, list) else video_path
                print(f'{video_path=}')
                # video_frames = get_video_frames(video_path, nframes=MAX_FRAMES)
                question = context.replace("<image>", "").replace("<video>", "").strip()
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": video_path,
                                # "max_pixels": 360 * 420,
                                # "fps": 1.0,
                                "nframes": MAX_FRAMES,
                            }, # 占位
                            {"type": "text", "text": question},
                        ],
                    }
                ]

                t1 = time.time()
                timings["Data Prep"] = (t1 - t0)

                if self.args.model_type == 'qwen25vl-7b':
                    text = self.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
                    inputs = self.processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                        **video_kwargs,
                    )
                elif self.args.model_type == 'qwen3vl-8b':
                    inputs = self.processor.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_dict=True,
                        return_tensors="pt"
                    )

                    # text = self.processor.apply_chat_template(
                    #     messages, tokenize=False, add_generation_prompt=True
                    # )
                    # image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
                    # inputs = self.processor(
                    #     text=[text],
                    #     images=image_inputs,
                    #     videos=video_inputs,
                    #     max_frames=MAX_FRAMES,
                    #     padding=True,
                    #     return_tensors="pt",
                    #     **video_kwargs,
                    # )

                    # text = self.processor.apply_chat_template(
                    #     messages, tokenize=False, add_generation_prompt=True
                    # )
                    # # 跳过process_vision_info，直接使用video_frames
                    # inputs = self.processor(
                    #     text=[text],
                    #     images=None,
                    #     videos=[video_frames],
                    #     padding=True,
                    #     return_tensors="pt",
                    #     max_pixels=224 * 224,
                    # )

                # update generate kwargs
                current_gen_kwargs = self.default_gen_kwargs.copy()
                if "max_new_tokens" in gen_kwargs:
                    current_gen_kwargs["max_new_tokens"] = gen_kwargs["max_new_tokens"]
                if "temperature" in gen_kwargs:
                    current_gen_kwargs["temperature"] = gen_kwargs["temperature"]
                    current_gen_kwargs["do_sample"] = True if gen_kwargs["temperature"] > 0 else False

                torch.cuda.synchronize() # 确保数据上传GPU完成
                t2 = time.time()
                timings["Preprocessing"] = (t2 - t1)
                
            
                # # generate output
                # print("Inputs:\n" + 70 * '-')
                # print(f"video: {video_path}")
                # print(f"question: {question}")
                # print(70 * '-' + "\n")
                # import pdb; pdb.set_trace()

                print(f'{inputs.input_ids.shape=}')

                with torch.inference_mode():
                    output_ids = self.model.generate(
                        inputs,
                        max_new_tokens=current_gen_kwargs["max_new_tokens"],
                        do_sample=current_gen_kwargs["do_sample"],
                        temperature=current_gen_kwargs["temperature"],
                        stop=None,
                        debug_mode=self.args.debug_mode,
                        cut_gen_len=self.args.cut_gen_len)
                    
                torch.cuda.synchronize()
                t3 = time.time()
                timings["Inference"] = (t3 - t2)

                # decode output
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output_ids)
                ]
                outputs = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                results.append(outputs[0])
                pbar.update(1)

                torch.cuda.synchronize()
                t4 = time.time()
                timings["Decoding"] = (t4 - t3)

                print_timing_stats(timings)
                # import pdb; pdb.set_trace()

                # print("Output:\n" + 70 * '-')
                # print(outputs[0])
                # print(70 * '-' + "\n")
                # import pdb; pdb.set_trace()
        
        except Exception as e:
            print(f"Error processing doc_id {doc_id}: {e}")
            results.append("")
        
        pbar.close()
        return results

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        raise NotImplementedError("generate_until_multi_round is not implemented for FlexGenLM.")

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        raise NotImplementedError("loglikelihood is not implemented for FlexGenLM.")
    
    def __del__(self):
        if hasattr(self, 'env'):
            self.env.close_copy_threads()