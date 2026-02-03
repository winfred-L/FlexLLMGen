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
import flexllmgen.utils as utils

import itertools
counter = itertools.count()

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


# def ensure_video_url(video: str) -> str:
#     prefixes = ['http://', 'https://', 'file://', 'data:video']
#     if any(video.startswith(prefix) for prefix in prefixes):
#         return video
#     if os.path.exists(video):
#         return 'file://' + video
#     raise ValueError(f'Invalid video: {video}')

# processor.apply_chat_template() 不接受file开头的路径, 所以直接使用绝对路径
def ensure_video_url(video: str) -> str:
    if os.path.exists(video):
        return video
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

        # print(messages) ### DEBUG

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )

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

        # print(response) ### DEBUG

        return response
    

    def __del__(self):
        if hasattr(self, 'env'):
            self.env.close_copy_threads()




def calculate_concentration(attention_tensor, epsilon=1e-9):
    """
    根据公式计算注意力向量的集中度分数 C。
    
    参数:
    attention_tensor (torch.Tensor): 一个一维的注意力权重张量 (通常总和不为 1, 函数内先进行归一化)。
    epsilon (float): 用于数值稳定性的极小常数，防止 log(0)。
    
    返回:
    float: 集中度分数 C，范围在 0 到 1 之间。
    """
    # 0. 归一化
    attn_sum = torch.sum(attention_tensor)
    if attn_sum > 0.05: # 筛选注意力权重累计和大于一定阈值的位置
        attention_tensor = attention_tensor / attn_sum
    else: # 如果总和太小，直接返回0
        return 0.0

    # 1. 获取 token 的数量 N
    N = attention_tensor.size(0)
    
    # 2. 计算熵 H = -sum(alpha * log2(alpha + epsilon))
    # 注意：为了与分母的 log2 保持底数一致，这里使用 torch.log2
    entropy = -torch.sum(attention_tensor * torch.log2(attention_tensor + epsilon))
    
    # 3. 计算最大可能熵 log2(N + epsilon)
    max_entropy = torch.log2(torch.tensor(N + epsilon, dtype=attention_tensor.dtype))
    
    # 4. 计算集中度分数 C = 1 - (H / max_entropy)
    concentration_score = 1 - (entropy / max_entropy)
    
    return concentration_score.item()


import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def draw_save_layer_head(
    matrix: torch.Tensor,
    title: str,
    output_path: str
):
    vmin = 0.0
    vmax = 1.0

    colors = [(1, 1, 1), (1, 0, 0)]  # white -> red
    cmap_name = 'white_to_red'
    white_to_red_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=256)

    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap=white_to_red_cmap, aspect='auto', vmin=vmin, vmax=vmax)
    plt.colorbar(label='Value')
    plt.title(title)
    plt.xlabel('head index')
    plt.ylabel('layer index')

    # Annotate each cell with the value (formatted to 2 decimal places)
    for i in range(matrix.size(0)):  # layers
        for j in range(matrix.size(1)):  # heads
            if matrix[i, j] != 0.0:
                plt.text(j, i, f'{matrix[i, j]:.2f}',
                            ha='center', va='center',
                            color='black' if (matrix[i, j] - vmin) / (vmax - vmin + 1e-8) < 0.5 else 'white',
                            fontsize=5)
            
    plt.tight_layout()
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    plt.savefig(f'{output_path}/{title}.png', dpi=150, bbox_inches='tight')
    plt.close()





# 用于eager分析注意力权重
class FlexGenTestLM(FlexGenLM):
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
        
        args.attn_impl = "eager"  ### attn analyse ###

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


    def _get_video_id(self, message):
        for s in message:
            if s['type'] == 'video':
                value = s['value']
                video_id = value.split('/')[-1]
                return video_id
        raise Exception('no video id found.')
    

    def generate_inner(self, message, dataset=None):
        model_type = self.args.model_type ### attn analyse ###
        video_id = self._get_video_id(message) ### attn analyse ###
        utils.total_attn_weight.clear() ### attn analyse ###


        messages = []
        messages.append({'role': 'user', 'content': self._prepare_content(message, dataset=dataset)})

        # print(messages) ### DEBUG

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )

        vision_info = self.model.get_video_info(inputs) ### attn analyse ###
        

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

        # print(response) ### DEBUG


        ### attn analyse ###
        max_step = utils.total_attn_weight.max_step
        index_ranges = vision_info.index_ranges

        if model_type == 'qwen3vl-8b':
            num_layer = 36
            num_head = 32
        else:
            raise NotImplementedError()
        shape = (num_layer, num_head, max_step-1)
        sum_result = torch.zeros(shape, dtype=float)
        con_result = torch.zeros(shape, dtype=float)

        for step in range(1, max_step):
            for layer in range(num_layer):
                attn_weight = utils.total_attn_weight.get(layer, step)
                
                for head in range(num_head):
                    vision_attn_weight = torch.cat([attn_weight[head][s : e+1] for s, e in index_ranges])

                    # count all vision attn weight
                    attn_weight_sum = torch.sum(vision_attn_weight)
                    sum_result[layer, head, step-1] = attn_weight_sum

                    # get concentration score
                    concentration = calculate_concentration(vision_attn_weight)
                    con_result[layer, head, step-1] = concentration

        doc_id = next(counter)
        output_path = f'/data1/lyc/flexllmgen_outputs/attn_weight/{dataset}_{model_type}/doc{doc_id}_{video_id}/'
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        torch.save(sum_result, f'{output_path}/sum_result.pt')
        torch.save(con_result, f'{output_path}/con_result.pt')

        sum_result_max, _ = torch.max(sum_result, dim=2)
        draw_save_layer_head(
            matrix = sum_result_max,
            title = 'sum_result_max',
            output_path = output_path,
        )

        score_result = sum_result * (1 + con_result)
        # set range into 0-1
        score_result_max_element = torch.max(score_result)
        score_result = score_result / score_result_max_element

        score_result_max, _ = torch.max(score_result, dim=2)
        draw_save_layer_head(
            matrix = score_result_max,
            title = 'score_result_max',
            output_path = output_path,
        )

        # import pdb; pdb.set_trace()

        return response