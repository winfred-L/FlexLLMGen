from dotenv import load_dotenv
load_dotenv(override=True)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import argparse
import os
import sys
from datetime import datetime

import torch
from transformers import AutoProcessor
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    print("qwen is not installed. Please install qwen-vl-utils to use this model.")

from flexllmgen.compression import CompressionConfig
from flexllmgen.pytorch_backend import (TorchDevice, TorchDisk,
    TorchMixedDevice, fix_recursive_import)
from flexllmgen.timer import timers
from flexllmgen.utils import ExecutionEnv, GB

fix_recursive_import()


from flexllmgen.policy import Policy

from flexllmgen.models.qwen25vl import Qwen25VLFlexLM
from flexllmgen.models.qwen25vl_config import get_qwen25vl_config
from flexllmgen.models.qwen3vl import Qwen3VLFlexLM
from flexllmgen.models.qwen3vl_config import get_qwen3vl_config



# def get_filename(args):
#     '''
#     根据输入参数（模型大小、Batch Size、卸载比例、压缩选项等）生成一个描述性的文件名，用于保存日志
#     '''
#     model_size = args.model.split('-')[-1]
#     percent = ""
#     for i in range(len(args.percent)):
#         percent += str(args.percent[i]) + "-"
#     filename = f"fo-{model_size}-gbs{args.gpu_batch_size}-" \
#                f"ngbs{args.num_gpu_batches}-" \
#                f"prompt{args.prompt_len}-" \
#                f"gen{args.gen_len}-percent-{percent}"
#     if args.cpu_cache_compute:
#         filename += "cpu-cache"
#     else:
#         filename += "gpu-cache"
#     if args.compress_weight:
#         filename += "-compw"
#     if args.compress_cache:
#         filename += "-compc"
#     return filename





def run_flexllmgen_qwen(args, video_path=None, question=None):
    num_prompts = args.num_gpu_batches * args.gpu_batch_size
    assert num_prompts == 1, "Only support batch size 1 for Qwen-VL now."
    assert not (args.compress_cache and args.attn_sparsity < 1.0), "Not implemented"

    # 1. 根据模型名称加载 Processor
    max_pixels: int = 16384*28*28
    min_pixels: int = 32*28*28
    if args.model_type == 'qwen25vl-7b':
        model_path = "/data/lyc/models/Qwen2.5-VL-7B-Instruct"
        # model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
        model_config = get_qwen25vl_config(args.model_type)
    elif args.model_type == 'qwen3vl-8b':
        model_path = "/data/lyc/models/Qwen3-VL-8B-Instruct"
        # model_path = "Qwen/Qwen3-VL-8B-Instruct"
        model_config = get_qwen3vl_config(args.model_type)
    processor = AutoProcessor.from_pretrained(
        model_path,
        max_pixels=max_pixels,
        min_pixels=min_pixels,
        use_fast=True
    )

    # 2. 准备输入数据
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
    
    # TODO: 合并
    if args.model_type == 'qwen25vl-7b':
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
    elif args.model_type == 'qwen3vl-8b':
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )

    warmup_inputs = inputs

    # 3. 准备执行环境
    gpu = TorchDevice(args.cuda_device)
    cpu = TorchDevice("cpu")
    disk = TorchDisk(os.getenv('OFFLOAD_DIR'))
    env = ExecutionEnv(gpu=gpu, cpu=cpu, disk=disk, mixed=TorchMixedDevice([gpu, cpu, disk]))

    # 3. 将命令行参数转换为 Policy 策略对象
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
    
    # 4. 模型初始化
    if args.model_type == 'qwen25vl-7b':
        model = Qwen25VLFlexLM(args.model_type, env, args.path, policy)
    elif args.model_type == 'qwen3vl-8b':
        model = Qwen3VLFlexLM(args.model_type, env, args.path, policy)


    # used for analyse attn weight
    if args.attn_impl == 'eager':
        vision_info = model.get_video_info(inputs)
        print(vision_info)


    # 5. 模型推理
    try:
        # Warmup：先跑一次短生成进行预热
        # print("warmup - generate")
        # with torch.inference_mode():
        #     output_ids = model.generate(
        #         warmup_inputs, max_new_tokens=2, verbose=args.verbose)

        # Benchmark：执行正式的生成任务，并记录时间
        # print("benchmark - generate")
        print("Inputs:\n" + 70 * '-')
        print(f"video: {video_path}")
        print(f"question: {question}")
        print(70 * '-' + "\n")

        timers("generate").reset()
        with torch.inference_mode():
            output_ids = model.generate(
                inputs,
                max_new_tokens=args.gen_len,
                do_sample=args.do_sample,
                temperature=args.temperature,
                stop=None,
                debug_mode=args.debug_mode,
                cut_gen_len=args.cut_gen_len)
        costs = timers("generate").costs
    finally:
        env.close_copy_threads()

    # # 7. 记录模型权重、kv cache、中间激活值的大小
    # encoder_weight_size = model_config.encoder_weight_bytes()
    # decoder_weight_size = model_config.decoder_weight_bytes()
    # weight_size = model_config.model_bytes()
    # print(f"model weight size: {weight_size/GB:.3f} GB")
    # print(f"  encoder weight size: {encoder_weight_size/GB:.3f} GB")
    # print(f"  decoder weight size: {decoder_weight_size/GB:.3f} GB")

    # 8. 记录推理输出
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output_ids)
    ]
    outputs = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    show_str = "\nOutputs:\n" + 70 * '-' + "\n"
    for i, output in enumerate(outputs):
        show_str += f"{output}\n"
        show_str += "-" * 70 + "\n"
    print(show_str)



def add_parser_arguments(parser):
    # ===== 模型设置 =====
    parser.add_argument("--model-type", type=str, default="qwen25vl-7b",
        choices=['qwen25vl-7b', 'qwen3vl-8b'],
        help="The model name.")
    parser.add_argument("--path", type=str, default="/data/lyc/models",
        help="The path to the model weights.") # TODO: 改为自动使用hf路径 #实际弃用，待删除
    parser.add_argument("--offload-dir", type=str, default="/data1/lyc/flexllmgen_offload_dir",
        help="The directory to offload tensors. ") # TODO: 实际弃用，待删除
    parser.add_argument("--sep-layer", type=bool, default=True) # 是否将 Attention 层和 MLP 层作为两个独立层处理
    parser.add_argument("--pin-weight", type=bool, default=True) # 是否将权重锁页到内存以加快 CPU 访问速度
    parser.add_argument("--overlap", type=bool, default=False) # 是否开启计算与数据传输重叠
    parser.add_argument("--percent", nargs="+", type=int,
        default=[100, 0, 100, 0, 100, 0],
        help="Six numbers. They are "
         "the percentage of weight on GPU, "
         "the percentage of weight on CPU, "
         "the percentage of attention cache on GPU, "
         "the percentage of attention cache on CPU, "
         "the percentage of activations on GPU, "
         "the percentage of activations on CPU")
    parser.add_argument("--attn-impl", type=str, default="flash_attn",
        choices=["eager", "flash_attn"],
        help="The attention implementation to use.") # eager will save attn_weight to file
    
    # ===== 推理设置 =====
    parser.add_argument("--cuda-device", type=str, default='cuda:0')
    parser.add_argument("--gen-len", type=int, default=50000) # 生成的最大新 Token 数量
    parser.add_argument("--do-sample", type=bool, default=False)
    parser.add_argument("--temperature", type=float, default=0.000001)
    parser.add_argument("--debug-mode", type=str, default=None, choices=["fewer_batch", "breakdown"])

    # ===== 稀疏设置 =====
    parser.add_argument("--do-sparse", type=bool, default=False)
    parser.add_argument("--threshold_S", type=float, default=0.0)
    parser.add_argument("--threshold_D", type=float, default=0.0)
    
    # ===== 固定设置 =====
    parser.add_argument("--gpu-batch-size", type=int, default=1)
    parser.add_argument("--num-gpu-batches", type=int, default=1)
    parser.add_argument("--cut-gen-len", type=int, default=None, help="Cut generation length for fast debugging.")
    parser.add_argument("--cpu-cache-compute", type=bool, default=False) # 是否在 CPU 上计算 Attention
    parser.add_argument("--attn-sparsity", type=float, default=1.0) # Attention 的稀疏度，1.0 代表稠密
    parser.add_argument("--compress-weight", type=bool, default=False,
        help="Whether to compress weight.")
    parser.add_argument("--compress-cache", type=bool, default=False,
        help="Whether to compress cache.")

    
    # ===== 弃置设置 =====
    # parser.add_argument("--prompt-len", type=int, default=512) # 输入提示的最大长度
    # parser.add_argument("--log-file", type=str, default="auto") # 日志文件名
    # parser.add_argument("--no-log", type=bool, default=True) # 不记录日志
    # parser.add_argument("--verbose", type=int, default=2) # 控制输出信息的详细程度
    

def print_args(args):
    # 格式化打印主要参数
    print(f"\n{'=' * 20} Configuration {'=' * 20}")
    for arg, value in vars(args).items():
        print(f"{arg:<20}: {value}")
    print(f"{'=' * 55}\n")


if __name__ == "__main__":
    # 参数设置
    parser = argparse.ArgumentParser()
    add_parser_arguments(parser)
    args = parser.parse_args()
    assert len(args.percent) == 6

    # args.model_type = "qwen25vl-7b"
    # args.model_type = "qwen3vl-8b"

    # args.percent = [100, 0, 0, 100, 100, 0] # all cache to cpu
    # args.percent = [100, 0, 0, 0, 100, 0] # all cache to disk


    ''' inference mode:
    |                      | args.debug_mode = None                 | args.debug_mode = 'breakdown'                |
    | -------------------- | -------------------------------------- | -------------------------------------------- |
    | args.overlap = False | generation_loop_normal()               | generation_loop_debug_normal()               |
    | args.overlap = True  | generation_loop_overlap_single_batch() | generation_loop_debug_overlap_single_batch() |
    '''
    # args.debug_mode = 'breakdown'
    # args.overlap = True

    # args.attn_impl = "eager"  # only for no sparse

    # only for generation_loop_overlap_single_batch()
    args.do_sparse = True


    video_path = "./test/video/28s.mp4"
    
    question = "Please describe this video in detail."

    # 重定向print()到log文件
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = f'./logs/{args.model_type}-{timestamp}.txt'
    log_file = open(output_file, 'w', encoding='utf-8')
    sys.stdout = log_file
    # sys.stderr = log_file

    # 打印参数设置
    print_args(args)

    # 项目入口
    if args.model_type == 'qwen25vl-7b':
        run_flexllmgen_qwen(args, video_path=video_path, question=question)
    elif args.model_type == 'qwen3vl-8b':
        run_flexllmgen_qwen(args, video_path=video_path, question=question)
    else:
        raise ValueError(f"Unsupported model: {args.model_type}")