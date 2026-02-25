# import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)
# warnings.simplefilter(action='ignore', category=UserWarning)

# import os
# MAX_FRAMES = 2048 # qwen2.5vl
# MAX_FRAMES = 4096 # qwen3vl
# os.environ["MAX_FRAMES"] = f"{MAX_FRAMES}"

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


# import torch
# from decord import VideoReader, gpu
# from PIL import Image
# import numpy as np

# def get_video_frames_as_pil(video_path, nframes):
#     # 1. GPU 解码
#     vr = VideoReader(video_path, ctx=gpu(0))
#     fps = vr.get_avg_fps()
#     total_frames = len(vr)
#     indices = np.linspace(0, total_frames - 1, nframes, dtype=int)
    
#     # 2. 获取 batch (GPU NDArray)
#     video_ndarray = vr.get_batch(indices) 
    
#     # 3. 搬回 CPU 并转换为 PIL (process_vision_info 对 PIL 的支持最稳)
#     # .asnumpy() 会将数据转为 [T, H, W, C] 的 uint8 数组
#     video_np = video_ndarray.asnumpy()
    
#     pil_frames = [Image.fromarray(frame) for frame in video_np]
#     return pil_frames, fps



import time
# def print_timing_stats(timings):
#     total_time = sum(timings.values())
#     print("\n" + "="*50)
#     print(f"{'Step':<20} | {'Time (s)':<10} | {'Ratio (%)':<10}")
#     print("-" * 50)
#     for step, t in timings.items():
#         ratio = (t / total_time) * 100
#         print(f"{step:<20} | {t:.4f}     | {ratio:.2f}%")
#     print("-" * 50)
#     print(f"{'Total':<20} | {total_time:.4f}     | 100.00%")
#     print("="*50 + "\n")




@register_model("flexgenlm")
class FlexGenLM(lmms):
    def __init__(self, **kwargs):
        super().__init__()

        # set up argument parser
        parser = argparse.ArgumentParser()
        add_parser_arguments(parser)
        args = parser.parse_args([])

        for k, v in kwargs.items():
            k = k.replace('-', '_')
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
            max_pixels: int = 100*1024*28*28 # 16384*28*28
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
            processor = AutoProcessor.from_pretrained(
                model_path,
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
                # torch.cuda.synchronize() # 确保之前任务完成
                # t0 = time.time()

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
                # print(f'{video_path=}')
                # video_frames, original_fps = get_video_frames_as_pil(video_path, nframes=MAX_FRAMES)
                question = context.replace("<image>", "").replace("<video>", "").strip()
                if self.args.model_type == 'qwen25vl-7b':
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "video",
                                    "video": video_path,
                                    # "max_pixels": 360 * 420,
                                    #"fps": 1.0,
                                    "nframes": 2048,
                                },
                                {"type": "text", "text": question},
                            ],
                        }
                    ]
                elif self.args.model_type == 'qwen3vl-8b':
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "video",
                                    "video": video_path,
                                    # restrict the resolution of individual frames in the video
                                    "min_pixels": 4 * 32 * 32,
                                    "max_pixels": 100 * 32 * 32, #256 * 32 * 32,
                                    # limit the total number of tokens in the video
                                    "total_pixels": 100 * 1024 * 32 * 32, # 224K tokens  #20480 * 32 * 32,
                                    # accept either `fps` or `nframes`
                                    #"fps": 2.0,
                                    "nframes": 2048,
                                },
                                {"type": "text", "text": question},
                            ],
                        }
                    ]

                # t1 = time.time()
                # print(f"t1-t0={t1-t0}")

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
                    text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    images, videos, video_kwargs = process_vision_info(messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)

                    # each video returns as (video_tensor, video_metadata)
                    # split the videos and according metadatas
                    if videos is not None:
                        videos, video_metadatas = zip(*videos)
                        videos, video_metadatas = list(videos), list(video_metadatas)
                    else:
                        video_metadatas = None

                    # since qwen-vl-utils has resize the images/videos, \
                    # we should pass do_resize=False to avoid duplicate operation in processor!
                    inputs = self.processor(
                        text=text,
                        images=images,
                        videos=videos,
                        video_metadata=video_metadatas,
                        return_tensors="pt",
                        do_resize=False, # avoid duplicate resizing
                        **video_kwargs
                    )

                # print(f'{inputs.input_ids.shape=}')

                # update generate kwargs
                current_gen_kwargs = self.default_gen_kwargs.copy()
                if "max_new_tokens" in gen_kwargs:
                    current_gen_kwargs["max_new_tokens"] = gen_kwargs["max_new_tokens"]
                if "temperature" in gen_kwargs:
                    current_gen_kwargs["temperature"] = gen_kwargs["temperature"]
                    current_gen_kwargs["do_sample"] = True if gen_kwargs["temperature"] > 0 else False

                # torch.cuda.synchronize()
                # t2 = time.time()
                # print(f"t2-t1={t2-t1}")
                
            
                # # generate output
                # print("Inputs:\n" + 70 * '-')
                # print(f"video: {video_path}")
                # print(f"question: {question}")
                # print(70 * '-' + "\n")
                # import pdb; pdb.set_trace()


                with torch.inference_mode():
                    output_ids = self.model.generate(
                        inputs,
                        max_new_tokens=current_gen_kwargs["max_new_tokens"],
                        do_sample=current_gen_kwargs["do_sample"],
                        temperature=current_gen_kwargs["temperature"],
                        stop=None,
                        debug_mode=self.args.debug_mode,
                        cut_gen_len=self.args.cut_gen_len)
                    
                # torch.cuda.synchronize()
                # t3 = time.time()
                # print(f"t3-t2={t3-t2}")

                # decode output
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output_ids)
                ]
                outputs = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                results.append(outputs[0])
                pbar.update(1)

                # torch.cuda.synchronize()
                # t4 = time.time()
                # print(f"t4-t3={t4-t3}")

                # print("Output:\n" + 70 * '-')
                # print(outputs[0])
                # print(70 * '-' + "\n")
                # import pdb; pdb.set_trace()

                import gc
                gc.collect()
                torch.cuda.empty_cache()
        
        except Exception as e:
            print(f"Error processing doc_id {doc_id}: {repr(e)}")
            import traceback
            traceback.print_exc()
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