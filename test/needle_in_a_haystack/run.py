from dotenv import load_dotenv
load_dotenv(override=True)

import argparse
import sys
import os
import json
from datetime import datetime
from typing import List, Tuple, Dict, Any
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


# ==================== 配置参数 ====================
dataset_json_path = "/data1/lyc/datasets/VNBench/VNBench-main-4try.json"  # 输入数据集路径
output_jsonl_path = "/data1/lyc/flexllmgen_outputs/VNBench/results.jsonl"  # 输出结果路径
model_path = "/data/lyc/models/Qwen3-VL-8B-Instruct"

# 视频处理参数
NFRAMES = 2048
MIN_PIXELS = 4 * 32 * 32
MAX_PIXELS = 100 * 32 * 32
TOTAL_PIXELS = 100 * 1024 * 32 * 32  # 约 100K tokens


# ==================== 工具函数 ====================
def load_dataset(json_path: str) -> List[dict]:
    """加载 JSON 格式的数据集"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples from {json_path}")
    return data


def save_result(result: dict, output_path: str):
    """将单个结果追加写入 JSONL 文件"""
    with open(output_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')


def generate_question_id(sample: dict, idx: int) -> str:
    """
    生成问题 ID
    格式：{video_filename}_{type}_{try}
    """
    video_path = sample.get('video', '')
    video_filename = os.path.basename(video_path).replace('.mp4', '')
    sample_type = sample.get('type', 'unknown')
    try_num = sample.get('try', 0)
    return f"{video_filename}_{sample_type}_{try_num}"


def extract_answer(output_text: str, options: List[str] = None) -> str:
    """
    从模型输出中提取答案选项（A/B/C/D）
    支持多种匹配模式
    """
    import re
    
    output_upper = output_text.upper()
    
    # 模式 1: 直接匹配选项字母 (A, B, C, D)
    # 匹配 "A", "(A)", "A.", "A:", "Answer: A", "The answer is A" 等
    patterns = [
        r'\b([A-D])\b',  # 单独字母
        r'\(([A-D])\)',  # (A)
        r'([A-D])[\.：:]',  # A. 或 A:
        r'ANSWER\s*[IS]?\s*([A-D])',  # ANSWER IS A
        r'OPTION\s*([A-D])',  # OPTION A
        r'CHOICE\s*([A-D])',  # CHOICE A
    ]
    
    for pattern in patterns:
        match = re.search(pattern, output_upper)
        if match:
            return match.group(1).upper()
    
    # 模式 2: 如果提供了 options，尝试匹配选项内容
    if options:
        for i, option in enumerate(options):
            if option.lower() in output_text.lower():
                return chr(ord('A') + i)
    
    # 默认返回 A（如果无法提取到有效选项）
    return 'A'


def clear_output_file(output_path: str):
    """清空输出文件（如果需要重新运行）"""
    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"Cleared existing output file: {output_path}")


# ==================== 主推理函数 ====================
def process_sample(
    sample: dict,
    model: Qwen3VLFlexLM,
    processor: AutoProcessor,
    args: argparse.Namespace,
    device: str = "cuda:0"
) -> Tuple[str, str, str]:
    """
    处理单个样本
    返回：(output_text, extracted_answer, prompt_text)
    """
    video_path = sample['video']
    video_path = video_path.replace("./VNBench/", "/data1/lyc/datasets/VNBench/VNBench-data/")
    question0 = sample['question']
    options = sample.get('options', None)
    question = f"{question0}\nA. {options[0]}\nB. {options[1]}\nC. {options[2]}\nD. {options[3]}\nAnswer with the option's letter from the given choices directly."

    print(f"Processing video: {video_path}")
    print(f"Question: {question}")
    
    # 构建消息
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "min_pixels": MIN_PIXELS,
                    "max_pixels": MAX_PIXELS,
                    "total_pixels": TOTAL_PIXELS,
                    # "nframes": NFRAMES,
                },
                {"type": "text", "text": question},
            ],
        }
    ]
    
    # 处理文本模板
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # 处理视觉信息
    images, videos, video_kwargs = process_vision_info(
        messages, 
        image_patch_size=16, 
        return_video_kwargs=True, 
        return_video_metadata=True
    )
    
    # 分离视频和元数据
    if videos is not None:
        videos, video_metadatas = zip(*videos)
        videos, video_metadatas = list(videos), list(video_metadatas)
    else:
        video_metadatas = None
    
    # 准备输入
    inputs = processor(
        text=text,
        images=images,
        videos=videos,
        video_metadata=video_metadatas,
        return_tensors="pt",
        do_resize=False,
        **video_kwargs
    )

    print(f'{inputs.input_ids.shape=}')
    
    # 生成
    with torch.inference_mode():
        output_ids = model.generate(
            inputs,
            max_new_tokens=args.gen_len,
            do_sample=args.do_sample,
            temperature=args.temperature,
            stop=None,
            debug_mode=args.debug_mode,
            cut_gen_len=args.cut_gen_len
        )
    
    # 解码输出
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )[0]
    
    # 提取答案
    extracted_answer = extract_answer(output_text, options)
    print(f"Extracted answer: {extracted_answer}")

    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    return output_text, extracted_answer, text


# ==================== 主程序 ====================
def main():
    # 设置设备
    device = "cuda:0"
    
    # 设置参数解析器
    parser = argparse.ArgumentParser()
    add_parser_arguments(parser)
    args = parser.parse_args([])
    
    # 初始化处理器
    print(f"Loading processor from {model_path}...")
    processor = AutoProcessor.from_pretrained(model_path)
    
    # 准备执行环境
    gpu = TorchDevice(args.cuda_device)
    cpu = TorchDevice("cpu")
    disk = TorchDisk(args.offload_dir)
    env = ExecutionEnv(gpu=gpu, cpu=cpu, disk=disk, mixed=TorchMixedDevice([gpu, cpu, disk]))
    
    policy = Policy(
        args.gpu_batch_size, args.num_gpu_batches,
        args.percent[0], args.percent[1],
        args.percent[2], args.percent[3],
        args.percent[4], args.percent[5],
        args.overlap, args.sep_layer, args.pin_weight,
        args.cpu_cache_compute, args.attn_sparsity,
        args.compress_weight, None,
        args.compress_cache, None,
        args.attn_impl,
        args.do_sparse, args.threshold_S, args.threshold_D
    )
    
    # 初始化模型
    print(f"Loading model from {args.path}...")
    model = Qwen3VLFlexLM(args.model_type, env, args.path, policy)
    
    
    # 加载数据集
    print(f"Loading dataset from {dataset_json_path}...")
    dataset = load_dataset(dataset_json_path)
    
    # 清空输出文件（如需增量添加可注释此行）
    clear_output_file(output_jsonl_path)
    
    # 推理循环
    print(f"Starting inference on {len(dataset)} samples...")
    print(f"Output will be saved to {output_jsonl_path}")
    print("=" * 60)
    
    success_count = 0
    error_count = 0
    
    for idx, sample in enumerate(tqdm(dataset, desc="Processing samples")):
        try:
            # 生成问题 ID
            question_id = generate_question_id(sample, idx)
            
            # 处理样本
            output_text, extracted_answer, prompt_text = process_sample(
                sample, model, processor, args, device
            )
            
            # 获取 ground truth
            gt_option = sample.get('gt_option', 'A')
            sample_type = sample.get('type', 'unknown')
            try_num = sample.get('try', 0)
            
            # 构建结果
            result = {
                "question_id": question_id,
                "pred": extracted_answer,
                "gt": gt_option,
                "type": sample_type,
                "try": try_num,
                "prompt": prompt_text,
                "output": output_text,
                "question": sample['question'],
                "video": sample['video']
            }
            
            # 保存结果
            save_result(result, output_jsonl_path)
            
            success_count += 1
            
            # 每 10 个样本打印一次进度和内存信息
            if (idx + 1) % 10 == 0:
                peak_mem = torch.cuda.max_memory_allocated(device)
                print(f"\n[Progress] {idx + 1}/{len(dataset)} samples processed")
                print(f"[Memory] CUDA peak: {peak_mem / 1024 / 1024 / 1024:.4f} GB")
                print(f"[Success/Error] {success_count}/{error_count}")
                print("=" * 60)
                
        except Exception as e:
            error_count += 1
            print(f"\n[ERROR] Sample {idx} failed: {str(e)}")
            
            # 保存错误信息
            error_result = {
                "question_id": generate_question_id(sample, idx),
                "pred": "ERROR",
                "gt": sample.get('gt_option', 'A'),
                "type": sample.get('type', 'unknown'),
                "try": sample.get('try', 0),
                "prompt": "",
                "output": f"ERROR: {str(e)}",
                "question": sample.get('question', ''),
                "video": sample.get('video', '')
            }
            save_result(error_result, output_jsonl_path)
            
            # 继续处理下一个样本
            continue
    
    # 最终统计
    print("\n" + "=" * 60)
    print("INFERENCE COMPLETED")
    print("=" * 60)
    print(f"Total samples: {len(dataset)}")
    print(f"Success: {success_count}")
    print(f"Errors: {error_count}")
    print(f"Output file: {output_jsonl_path}")
    
    # 最终内存信息
    peak_mem = torch.cuda.max_memory_allocated(device)
    print(f"CUDA peak memory: {peak_mem / 1024 / 1024 / 1024:.4f} GB")


if __name__ == "__main__":
    main()