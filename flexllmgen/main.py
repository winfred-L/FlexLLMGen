import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import argparse

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
    if args.model == 'qwen25vl-7b':
        model_path = "/data/lyc/models/Qwen2.5-VL-7B-Instruct"
        max_pixels: int = 16384*28*28
        min_pixels: int = 32*28*28
        processor = AutoProcessor.from_pretrained(
            model_path,
            max_pixels=max_pixels,
            min_pixels=min_pixels,
            use_fast=True
        )
        model_config = get_qwen25vl_config(args.model)
    elif args.model == 'qwen3vl-8b':
        model_path = "/data/lyc/models/Qwen3-VL-8B-Instruct"
        processor = AutoProcessor.from_pretrained(model_path)
        model_config = get_qwen3vl_config(args.model)

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
    
    if args.model == 'qwen25vl-7b':
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
    elif args.model == 'qwen3vl-8b':
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
    
    # 4. 模型初始化
    if args.model == 'qwen25vl-7b':
        model = Qwen25VLFlexLM(args.model, env, args.path, policy)
    elif args.model == 'qwen3vl-8b':
        model = Qwen3VLFlexLM(args.model, env, args.path, policy)

    # 5. 模型推理
    try:
        # Warmup：先跑一次短生成进行预热
        # print("warmup - generate")
        # with torch.inference_mode():
        #     output_ids = model.generate(
        #         warmup_inputs, max_new_tokens=2, verbose=args.verbose)

        # Benchmark：执行正式的生成任务，并记录时间
        # print("benchmark - generate")
        print("Inputs:\n" + 70 * '-' + "\n")
        print(f"video: {video_path}")
        print(f"question: {question}")

        timers("generate").reset()
        with torch.inference_mode():
            output_ids = model.generate(
                inputs,
                max_new_tokens=args.max_gen_len,
                do_sample=args.do_sample,
                temperature=args.temperature,
                stop=None,
                debug_mode=args.debug_mode,
                cut_gen_len=args.cut_gen_len)
        costs = timers("generate").costs
    finally:
        env.close_copy_threads()

    # 7. 记录推理输出
    outputs = processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    show_str = "Outputs:\n" + 70 * '-' + "\n"
    for i, output in enumerate(outputs):
        show_str += f"{i}: {output}\n"
        show_str += "-" * 70 + "\n"
    if args.verbose >= 2:
        print(show_str)
        
    # 8. 记录硬件性能统计
    gpu.print_stats()
    cpu.print_stats()
    # disk.print_stats() # NotImplemented

    # 9. 记录模型权重、kv cache、中间激活值的大小
    encoder_weight_size = model_config.encoder_weight_bytes()
    decoder_weight_size = model_config.decoder_weight_bytes()
    weight_size = model_config.model_bytes()

    # TODO: 记录推理过程中实时的内存峰值

    # cache_size = model_config.cache_bytes(num_prompts, prompt_len + gen_len)
    # hidden_size = model_config.hidden_bytes(num_prompts, prompt_len + gen_len)
    
    # print(f"model weight size: {weight_size/GB:.3f} GB, "
    #       f"kv cache size:     {cache_size/GB:.3f} GB, "
    #       f"hidden state size: {hidden_size/GB:.3f} GB")



def add_parser_arguments(parser):
    # ===== 模型设置 =====
    parser.add_argument("--model", type=str, default="qwen25vl-7b",
        choices=['opt-1.3b', 'qwen25vl-7b', 'qwen3vl-8b'],
        help="The model name.")
    parser.add_argument("--path", type=str, default="/data/lyc/models",
        help="The path to the model weights.")
    parser.add_argument("--offload-dir", type=str, default="/data1/lyc/flexllmgen_offload_dir",
        help="The directory to offload tensors. ") # disk 卸载目录
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
    
    
    # ===== 推理设置 =====
    parser.add_argument("--cuda-device", type=str, default='cuda:0')
    parser.add_argument("--max-gen-len", type=int, default=512)
    parser.add_argument("--do-sample", type=bool, default=False)
    parser.add_argument("--temperature", type=float, default=0.000001)
    parser.add_argument("--debug-mode", type=str, default=None, choices=["fewer_batch", "breakdown"])
    

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
    # parser.add_argument("--gen-len", type=int, default=512) # 生成的最大新 Token 数量
    # parser.add_argument("--log-file", type=str, default="auto") # 日志文件名
    # parser.add_argument("--no-log", type=bool, default=True) # 不记录日志
    # parser.add_argument("--verbose", type=int, default=2) # 控制输出信息的详细程度
    


if __name__ == "__main__":
    # 参数设置
    parser = argparse.ArgumentParser()
    add_parser_arguments(parser)
    args = parser.parse_args()
    assert len(args.percent) == 6

    # args.model = "opt-1.3b"
    args.model = "qwen25vl-7b"
    # args.model = "qwen3vl-8b"

    args.debug_mode = 'breakdown'

    video_path = "/data/lyc/datasets/Video-MME/video/ZHWZf1Z4B5k.mp4" #28s
    # video_path = "/data/lyc/datasets/Video-MME/video/zNxi2s36tS0.mp4" #43s
    # video_path = "/data/lyc/datasets/Video-MME/video/Z-rHofd6g2Q.mp4" #66s
    question = "Please describe this video in detail."

    # 项目入口
    if args.model == 'qwen25vl-7b':
        run_flexllmgen_qwen(args, video_path=video_path, question=question)
    elif args.model == 'qwen3vl-8b':
        run_flexllmgen_qwen(args, video_path=video_path, question=question)
    else:
        raise ValueError(f"Unsupported model: {args.model}")
