"""
Usage:
python3 -m flexllmgen.flex_opt --model facebook/opt-1.3b --gpu-batch-size 32 --percent 100 0 100 0 100 0
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import argparse

import os
import pickle
import time
from typing import Union, List, Optional

import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoProcessor, AutoConfig
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    print("qwen is not installed. Please install qwen-vl-utils to use this model.")

from flexllmgen.compression import CompressionConfig
from flexllmgen.pytorch_backend import (TorchDevice, TorchDisk, TorchLink,
    TorchMixedDevice, DeviceType, general_copy, fix_recursive_import)
from flexllmgen.timer import timers
from flexllmgen.utils import (Task, ExecutionEnv, GB, T, ValueHolder,
    array_1d, array_2d, array_3d, str2bool, project_decode_latency,
    torch_mem_stats, torch_dtype_to_np_dtype, write_benchmark_log,
    read_benchmark_log)

fix_recursive_import()


from flexllmgen.policy import Policy, DUMMY_WEIGHT
from flexllmgen.models.opt_config import get_opt_config
from flexllmgen.models.qwen25vl_config import get_qwen25vl_config
from flexllmgen.models.opt import OptLM
from flexllmgen.models.qwen25vl import Qwen25VLLM




def get_filename(args):
    '''
    根据输入参数（模型大小、Batch Size、卸载比例、压缩选项等）生成一个描述性的文件名，用于保存日志
    '''
    model_size = args.model.split('-')[-1]
    percent = ""
    for i in range(len(args.percent)):
        percent += str(args.percent[i]) + "-"
    filename = f"fo-{model_size}-gbs{args.gpu_batch_size}-" \
               f"ngbs{args.num_gpu_batches}-" \
               f"prompt{args.prompt_len}-" \
               f"gen{args.gen_len}-percent-{percent}"
    if args.cpu_cache_compute:
        filename += "cpu-cache"
    else:
        filename += "gpu-cache"
    if args.compress_weight:
        filename += "-compw"
    if args.compress_cache:
        filename += "-compc"
    return filename


def get_test_inputs(prompt_len, num_prompts, tokenizer):
    '''
    生成测试用的输入 Token IDs（将 "Paris is the capital city of" padding 到指定长度）
    '''
    prompts = ["Paris is the capital city of"]
    input_ids = tokenizer(prompts, padding="max_length",
                          max_length=prompt_len).input_ids
    return (input_ids[0],) * num_prompts


def run_flexllmgen_opt(args):
    # 1. 根据模型名称加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("/data/lyc/models/opt-1.3b", padding_side="left")

    # 2. 准备输入数据和执行环境
    num_prompts = args.num_gpu_batches * args.gpu_batch_size
    prompt_len, gen_len, cut_gen_len = args.prompt_len, args.gen_len, args.cut_gen_len
    warmup_inputs = get_test_inputs(32, num_prompts, tokenizer)
    inputs = get_test_inputs(prompt_len, num_prompts, tokenizer)

    gpu = TorchDevice("cuda:0")
    cpu = TorchDevice("cpu")
    disk = TorchDisk(args.offload_dir)
    env = ExecutionEnv(gpu=gpu, cpu=cpu, disk=disk, mixed=TorchMixedDevice([gpu, cpu, disk]))

    # 3. 将命令行参数转换为 Policy 策略对象
    policy = Policy(args.gpu_batch_size, args.num_gpu_batches,
                    args.percent[0], args.percent[1],
                    args.percent[2], args.percent[3],
                    args.percent[4], args.percent[5],
                    args.overlap, args.sep_layer, args.pin_weight,
                    args.cpu_cache_compute, args.attn_sparsity,
                    args.compress_weight,
                    CompressionConfig(num_bits=4, group_size=64,
                                      group_dim=0, symmetric=False),
                    args.compress_cache,
                    CompressionConfig(num_bits=4, group_size=64,
                                      group_dim=2, symmetric=False))
    assert not (args.compress_cache and args.attn_sparsity < 1.0), "Not implemented"

    # 4. 预估模型权重、kv cache、中间激活值的大小，模型初始化
    model_config = get_opt_config(args.model)
    weight_size = model_config.model_bytes()
    cache_size = model_config.cache_bytes(num_prompts, prompt_len + gen_len)
    hidden_size = model_config.hidden_bytes(num_prompts, prompt_len + gen_len)
    print(f"model weight size: {weight_size/GB:.3f} GB, "
          f"kv cache size: {cache_size/GB:.3f} GB, "
          f"hidden size (prefill): {hidden_size/GB:.3f} GB")

    print("init weight...")
    model = OptLM(args.model, env, args.path, policy)
    # 注：model_config并没有用在OptLM的初始化中，只是用来计算内存占用

    try:
        # 5. Warmup：先跑一次短生成进行预热
        print("warmup - generate")
        output_ids = model.generate(
            warmup_inputs, max_new_tokens=1, verbose=args.verbose)

        # 6. Benchmark：执行正式的生成任务，并记录时间
        print("benchmark - generate")
        timers("generate").reset()
        output_ids = model.generate(
            inputs, max_new_tokens=args.gen_len,
            debug_mode=args.debug_mode, cut_gen_len=cut_gen_len, verbose=args.verbose)
        costs = timers("generate").costs
    finally:
        env.close_copy_threads()

    # 7. 性能统计与日志记录
    prefill_latency = costs[0]
    prefill_throughput = num_prompts * prompt_len / prefill_latency
    if cut_gen_len:  # project latency of cut_gen_len to gen_len
        decode_latency = project_decode_latency(costs, prompt_len, gen_len)
    else:
        decode_latency = sum(costs[1:])
    decode_throughput = num_prompts * (gen_len - 1) / max(decode_latency, 1e-10)
    num_generated_tokens = num_prompts * gen_len
    total_latency = prefill_latency + decode_latency
    total_throughput = num_generated_tokens / total_latency
    _, gpu_peak_mem = gpu.mem_stats()
    _, cpu_peak_mem = cpu.mem_stats()

    if DUMMY_WEIGHT not in args.path:
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        show_str = "Outputs:\n" + 70 * '-' + "\n"
        for i in [0, len(outputs)-1]:
            show_str += f"{i}: {outputs[i]}\n"
            show_str += "-" * 70 + "\n"
        if args.verbose >= 2:
            print(show_str)

    gpu.print_stats()
    cpu.print_stats()
    projected = bool(args.debug_mode or cut_gen_len)

    if not args.no_log:
        if args.log_file == "auto":
            filename = get_filename(args) + ".log"
        else:
            filename = args.log_file

        log_str = write_benchmark_log(filename,
            model_config.model_bytes(), cache_size, hidden_size,
            gpu_peak_mem, projected, prefill_latency, prefill_throughput,
            decode_latency, decode_throughput, total_latency, total_throughput)
    
        if args.verbose >= 1:
            print(log_str)


def run_flexllmgen_qwen25vl(args):
    model_path = "/data/lyc/models/Qwen2.5-VL-7B-Instruct"
    # 1. 根据模型名称加载 Processor
    max_pixels: int = 16384*28*28
    min_pixels: int = 32*28*28
    processor = AutoProcessor.from_pretrained(
        model_path,
        max_pixels=max_pixels,
        min_pixels=min_pixels,
        use_fast=True
    )
    config = AutoConfig.from_pretrained(
        model_path, 
        trust_remote_code=True
    )

    # 2. 准备输入数据和执行环境
    num_prompts = args.num_gpu_batches * args.gpu_batch_size
    assert num_prompts == 1, "Only support batch size 1 for Qwen2.5-VL now."
    prompt_len, gen_len, cut_gen_len = args.prompt_len, args.gen_len, args.cut_gen_len

    video_path = "/data/lyc/datasets/Video-MME/video/ZHWZf1Z4B5k.mp4"
    question = "Please describe this video in detail."
    video_fps = 1.0
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": 360 * 420,
                    "fps": video_fps,
                },
                {"type": "text", "text": question},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )
    warmup_inputs = inputs

    gpu = TorchDevice("cuda:0")
    cpu = TorchDevice("cpu")
    disk = TorchDisk(args.offload_dir)
    env = ExecutionEnv(gpu=gpu, cpu=cpu, disk=disk, mixed=TorchMixedDevice([gpu, cpu, disk]))

    # 3. 将命令行参数转换为 Policy 策略对象
    policy = Policy(args.gpu_batch_size, args.num_gpu_batches,
                    args.percent[0], args.percent[1],
                    args.percent[2], args.percent[3],
                    args.percent[4], args.percent[5],
                    args.overlap, args.sep_layer, args.pin_weight,
                    args.cpu_cache_compute, args.attn_sparsity,
                    args.compress_weight,
                    CompressionConfig(num_bits=4, group_size=64,
                                      group_dim=0, symmetric=False),
                    args.compress_cache,
                    CompressionConfig(num_bits=4, group_size=64,
                                      group_dim=2, symmetric=False))
    assert not (args.compress_cache and args.attn_sparsity < 1.0), "Not implemented"

    # 4. 预估模型权重、kv cache、中间激活值的大小，模型初始化
    model_config = get_qwen25vl_config(args.model)
    weight_size = model_config.model_bytes()
    cache_size = model_config.cache_bytes(num_prompts, prompt_len + gen_len)
    hidden_size = model_config.hidden_bytes(num_prompts, prompt_len + gen_len)
    print(f"model weight size: {weight_size/GB:.3f} GB, "
          f"kv cache size: {cache_size/GB:.3f} GB, "
          f"hidden size (prefill): {hidden_size/GB:.3f} GB")

    print("init weight...")
    model = Qwen25VLLM(args.model, env, args.path, policy)

    try:
        # # 5. Warmup：先跑一次短生成进行预热
        # print("warmup - generate")
        # with torch.inference_mode():
        #     output_ids = model.generate(
        #         warmup_inputs, max_new_tokens=2, verbose=args.verbose)

        # 6. Benchmark：执行正式的生成任务，并记录时间
        print("benchmark - generate")
        timers("generate").reset()
        with torch.inference_mode():
            output_ids = model.generate(
                inputs, max_new_tokens=args.gen_len,
                debug_mode=args.debug_mode, cut_gen_len=cut_gen_len, verbose=args.verbose)
        costs = timers("generate").costs
    finally:
        env.close_copy_threads()

    # 7. 性能统计与日志记录
    prefill_latency = costs[0]
    prefill_throughput = num_prompts * prompt_len / prefill_latency
    if cut_gen_len:  # project latency of cut_gen_len to gen_len
        decode_latency = project_decode_latency(costs, prompt_len, gen_len)
    else:
        decode_latency = sum(costs[1:])
    decode_throughput = num_prompts * (gen_len - 1) / max(decode_latency, 1e-10)
    num_generated_tokens = num_prompts * gen_len
    total_latency = prefill_latency + decode_latency
    total_throughput = num_generated_tokens / total_latency
    _, gpu_peak_mem = gpu.mem_stats()
    _, cpu_peak_mem = cpu.mem_stats()

    if DUMMY_WEIGHT not in args.path:
        outputs = processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        show_str = "Outputs:\n" + 70 * '-' + "\n"
        for i, output in enumerate(outputs):
            show_str += f"{i}: {output}\n"
            show_str += "-" * 70 + "\n"
        if args.verbose >= 2:
            print(show_str)

    gpu.print_stats()
    cpu.print_stats()
    projected = bool(args.debug_mode or cut_gen_len)

    if not args.no_log:
        if args.log_file == "auto":
            filename = get_filename(args) + ".log"
        else:
            filename = args.log_file

        log_str = write_benchmark_log(filename,
            model_config.model_bytes(), cache_size, hidden_size,
            gpu_peak_mem, projected, prefill_latency, prefill_throughput,
            decode_latency, decode_throughput, total_latency, total_throughput)
    
        if args.verbose >= 1:
            print(log_str)




def add_parser_arguments(parser):
    parser.add_argument("--model", type=str, default="opt-1.3b",
        choices=['opt-1.3b', 'qwen25vl-7b', 'qwen3vl-8b'],
        help="The model name.")
    parser.add_argument("--path", type=str, default="/data/lyc/models",
        help="The path to the model weights. If there are no cached weights, "
             "FlexLLMGen will automatically download them from HuggingFace.")
    parser.add_argument("--offload-dir", type=str, default="/data1/lyc/flexllmgen_offload_dir",
        help="The directory to offload tensors. ") # disk 卸载目录
    parser.add_argument("--prompt-len", type=int, default=512) # 输入提示的长度
    parser.add_argument("--gen-len", type=int, default=128) # 生成的最大新 Token 数量
    parser.add_argument("--cut-gen-len", type=int,
        help="Cut generation length for fast debugging.")
    parser.add_argument("--debug-mode", type=str, default=None,
        choices=["fewer_batch", "breakdown"])
    parser.add_argument("--overlap", type=str2bool, nargs='?', # 是否开启 I/O 和计算的重叠
        const=True, default=False)
    parser.add_argument("--gpu-batch-size", type=int, default=1)
    parser.add_argument("--num-gpu-batches", type=int, default=1)
    parser.add_argument("--percent", nargs="+", type=int,
        default=[100, 0, 100, 0, 100, 0],
        help="Six numbers. They are "
         "the percentage of weight on GPU, "
         "the percentage of weight on CPU, "
         "the percentage of attention cache on GPU, "
         "the percentage of attention cache on CPU, "
         "the percentage of activations on GPU, "
         "the percentage of activations on CPU")
    parser.add_argument("--sep-layer", type=str2bool, nargs='?', # 是否将 Attention 层和 MLP 层作为两个独立层处理
        const=True, default=True)
    # str2bool 将命令行传入的字符串（如 "True", "False", "yes", "no", "1", "0"）转换成 Python 的布尔值 (True 或 False)
    # nargs='?' 表示该参数接受 0 个或 1 个 值
    parser.add_argument("--pin-weight", type=str2bool, nargs="?", # 是否将权重锁页到内存以加快 CPU 访问速度
        const=True, default=True)
    parser.add_argument("--cpu-cache-compute", type=bool, default=False) # 是否在 CPU 上计算 Attention
    parser.add_argument("--attn-sparsity", type=float, default=1.0) # Attention 的稀疏度，1.0 代表稠密
    parser.add_argument("--compress-weight", type=bool, default=False,
        help="Whether to compress weight.")
    parser.add_argument("--compress-cache", type=bool, default=False,
        help="Whether to compress cache.")


    parser.add_argument("--log-file", type=str, default="auto") # 日志文件名
    parser.add_argument("--no-log", type=bool, default=True) # 不记录日志
    parser.add_argument("--verbose", type=int, default=2) # 控制输出信息的详细程度


if __name__ == "__main__":
    # 参数设置
    parser = argparse.ArgumentParser()
    add_parser_arguments(parser)
    args = parser.parse_args()
    assert len(args.percent) == 6

    # args.model = "opt-1.3b"
    args.model = "qwen25vl-7b"

    # 项目入口
    if args.model == "opt-1.3b":
        run_flexllmgen_opt(args)
    elif args.model == 'qwen25vl-7b':
        run_flexllmgen_qwen25vl(args)
    elif args.model == 'qwen3vl-8b':
        pass # TODO
    else:
        raise ValueError(f"Unsupported model: {args.model}")
