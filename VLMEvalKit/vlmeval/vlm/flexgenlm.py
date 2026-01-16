import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import os
import argparse
import sys
from datetime import datetime
from typing import List, Tuple
from tqdm import tqdm
import copy as cp

import torch
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer
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

# VLMEvalKit imports
from .base import BaseModel
from ..smp import isimg, listinstr
from ..dataset import DATASET_TYPE



def ensure_image_url(image: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:image']
    if any(image.startswith(prefix) for prefix in prefixes):
        return image
    if os.path.exists(image):
        return 'file://' + image
    raise ValueError(f'Invalid image: {image}')


def ensure_video_url(video: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:video']
    if any(video.startswith(prefix) for prefix in prefixes):
        return video
    if os.path.exists(video):
        return 'file://' + video
    raise ValueError(f'Invalid video: {video}')


class FlexGenLM(BaseModel):
    INSTALL_REQ = False # installation requirement not needed
    INTERLEAVE = False # interleaved image and text input not supported
    VIDEO_LLM = True # video input supported

    def __init__(self, model_path=None, **kwargs):
        assert model_path is None # set via kwargs
        
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
        self.args = args

        # initialize processor
        self.max_pixels = 16384*28*28
        self.min_pixels = 32*28*28
        self.total_pixels = None
        self.fps = 1.0
        self.nframe = None
        self.FRAME_FACTOR = None

        if args.model_type == 'qwen25vl-7b':
            model_path = "/data/lyc/models/Qwen2.5-VL-7B-Instruct"
            model_config = get_qwen25vl_config(args.model_type)
        elif args.model_type == 'qwen3vl-8b':
            model_path = "/data/lyc/models/Qwen3-VL-8B-Instruct"
            model_config = get_qwen3vl_config(args.model_type)
        
        self.model_path = model_path
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            max_pixels=self.max_pixels,
            min_pixels=self.min_pixels,
            use_fast=True
        )

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

        self.model = model
        torch.cuda.empty_cache()


    def _prepare_content(self, inputs: list[dict[str, str]], dataset: str | None = None) -> list[dict[str, str]]:
        content = []
        for s in inputs:
            if s['type'] == 'image':
                raise NotImplementedError('FlexGenLM does not support image input.')
                # item = {'type': 'image', 'image': ensure_image_url(s['value'])}
                # if dataset == 'OCRBench':
                #     item['min_pixels'] = 10 * 10 * 32 * 32
                #     warnings.warn(f"OCRBench dataset uses custom min_pixels={item['min_pixels']}")
                #     if self.max_pixels is not None:
                #         item['max_pixels'] = self.max_pixels
                # else:
                #     if self.min_pixels is not None:
                #         item['min_pixels'] = self.min_pixels
                #     if self.max_pixels is not None:
                #         item['max_pixels'] = self.max_pixels
                # if self.total_pixels is not None:
                #     item['total_pixels'] = self.total_pixels
                # for key in ['min_pixels', 'max_pixels', 'total_pixels', 'resized_height', 'resized_width']:
                #     if key in s and s[key] is not None:
                #         item[key] = s[key]
            elif s['type'] == 'video':
                value = s['value']
                if isinstance(value, list):
                    item = {
                        'type': 'video',
                        'video': [ensure_image_url(v) for v in value],
                    }
                else:
                    item = {'type': 'video', 'video': ensure_video_url(value)}
                if self.min_pixels is not None:
                    item['min_pixels'] = self.min_pixels
                if self.max_pixels is not None:
                    item['max_pixels'] = self.max_pixels
                if self.total_pixels is not None:
                    item['total_pixels'] = self.total_pixels
                for key in ['resized_height', 'resized_width', 'fps', 'nframes', 'sample_fps']:
                    if key in s and s[key] is not None:
                        item[key] = s[key]
                if not isinstance(value, list):
                    if self.fps is not None and 'fps' not in item:
                        item['fps'] = self.fps
                    elif self.nframe is not None and 'nframes' not in item:
                        import cv2
                        video = cv2.VideoCapture(s['value'])
                        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                        video.release()
                        if frame_count < self.nframe:
                            new_frame_count = frame_count // self.FRAME_FACTOR * self.FRAME_FACTOR
                            print(f"use {new_frame_count} for {s['value']}")
                            item['nframes'] = new_frame_count
                        else:
                            item['nframes'] = self.nframe
            elif s['type'] == 'audio':
                item = {'type': 'audio', 'audio': s['value']}
            elif s['type'] == 'text':
                item = {'type': 'text', 'text': s['value']}
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
            content.append(item)
        return content

    def generate_inner(self, message, dataset=None):
        messages = []
        messages.append({'role': 'user', 'content': self._prepare_content(message, dataset=dataset)})

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )

        import pdb; pdb.set_trace()

        try:
            inputs = inputs.to(self.model.device)
            if hasattr(self.model, 'dtype'):
                inputs = inputs.to(self.model.dtype)
        except Exception:
            inputs = inputs.to('cuda')


        # inference
        generated_ids = self.model.generate(
            inputs,
            max_new_tokens=self.args.gen_len,
            do_sample=self.args.do_sample,
            temperature=self.args.temperature,
            stop=None,
            debug_mode=self.args.debug_mode,
            cut_gen_len=self.args.cut_gen_len,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        out = self.processor.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        response = out[0]
        return response
    

    def __del__(self):
        if hasattr(self, 'env'):
            self.env.close_copy_threads()