from dotenv import load_dotenv
load_dotenv(override=True)
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True,max_split_size_mb:128'

import argparse
import sys

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
OPEN_COT = True
DO_SPARSE = False

output_file = 'MLVU_needle_over10min'
if OPEN_COT:
    output_file += '_cot'
output_file += '-qwen3vl-8b'
if DO_SPARSE:
    output_file += '-no_vid'

# 修改为 MLVU_needle 数据集路径
dataset_json_path = "/data1/lyc/datasets/MLVU/MLVU/json/2_needle.json"
output_jsonl_path = f"/data1/lyc/flexllmgen_outputs/MLVU_needle/{output_file}.jsonl"
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
    生成问题 ID（适配 MLVU 格式）
    格式：{video_filename}_{question_type}
    """
    video_path = sample.get('video', '')
    video_filename = os.path.basename(video_path).replace('.mp4', '')
    question_type = sample.get('question_type', 'unknown')
    return f"{video_filename}_{question_type}"

def extract_answer(output_text: str, candidates: List[str] = None, ground_truth: str = None) -> str:
    """
    从模型输出中提取答案
    支持匹配 candidates 内容或直接返回提取的文本
    """
    import re
    output_lower = output_text.lower()
    
    # 模式 1: 尝试匹配 candidates 中的内容
    if candidates:
        for i, candidate in enumerate(candidates):
            if candidate.lower() in output_lower:
                return candidate
    
    # 模式 2: 尝试提取引号内的内容
    quote_patterns = [
        r'["\']([^"\']+)["\']',
        r'答案 [是:：]\s*["\']?([^"\']+)["\']?',
        r'answer\s*[is:]+\s*["\']?([^"\']+)["\']?',
    ]
    
    for pattern in quote_patterns:
        match = re.search(pattern, output_text, re.IGNORECASE)
        if match:
            extracted = match.group(1).strip()
            # 检查是否与某个 candidate 匹配
            if candidates:
                for candidate in candidates:
                    if candidate.lower() in extracted.lower() or extracted.lower() in candidate.lower():
                        return candidate
            return extracted
    
    # 模式 3: 直接返回输出的第一行（去除特殊标记）
    lines = output_text.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line and not line.startswith(('#', '*', '-')):
            if candidates:
                for candidate in candidates:
                    if candidate.lower() in line.lower():
                        return candidate
            return line
    
    # 默认返回第一个 candidate
    return candidates[0] if candidates else "A"

# COT version
def extract_answer_cot(output_text: str, candidates: List[str] = None, ground_truth: str = None) -> str:
    """
    从 COT 格式输出中提取答案
    支持匹配 "Best Option: (A)" 格式，然后映射到 candidates 完整文本
    """
    import re
    output_upper = output_text.upper()
    
    # ========== 模式 1: 匹配 "Best Option" 字段中的选项字母 ==========
    # 匹配 "Best Option: (A)" 或 "Best Option: A" 等格式
    best_option_patterns = [
        r'BEST\s*OPTION\s*[:：]?\s*\(([A-D])\)',  # Best Option: (A)
        r'BEST\s*OPTION\s*[:：]?\s*([A-D])\b',    # Best Option: A
        r'3\.\s*.*?\(([A-D])\)',                   # 3. ... (A)
        r'3\.\s*.*?\b([A-D])\b',                   # 3. ... A
    ]
    
    for pattern in best_option_patterns:
        match = re.search(pattern, output_upper)
        if match:
            option_letter = match.group(1).upper()
            # 映射到 candidates 完整文本
            if candidates and len(candidates) >= ord(option_letter) - ord('A') + 1:
                idx = ord(option_letter) - ord('A')
                return candidates[idx]
            return option_letter
    
    # ========== 模式 2: 匹配其他常见选项格式 ==========
    general_patterns = [
        r'\b([A-D])\b',           # 单独字母
        r'\(([A-D])\)',           # (A)
        r'([A-D])[\.：:]',        # A. 或 A:
        r'ANSWER\s*[IS]?\s*([A-D])',  # ANSWER IS A
        r'OPTION\s*([A-D])',      # OPTION A
        r'CHOICE\s*([A-D])',      # CHOICE A
        r'THE\s*CORRECT\s*(?:OPTION|ANSWER)\s*[IS]?\s*([A-D])',  # THE CORRECT OPTION IS A
    ]
    
    for pattern in general_patterns:
        match = re.search(pattern, output_upper)
        if match:
            option_letter = match.group(1).upper()
            if candidates and len(candidates) >= ord(option_letter) - ord('A') + 1:
                idx = ord(option_letter) - ord('A')
                return candidates[idx]
            return option_letter
    
    # ========== 模式 3: 直接匹配 candidates 内容 ==========
    if candidates:
        for candidate in candidates:
            if candidate.lower() in output_text.lower():
                return candidate
    
    # ========== 模式 4: 提取引号内的内容 ==========
    quote_patterns = [
        r'["\']([^"\']+)["\']',
        r'答案 [是:：]\s*["\']?([^"\']+)["\']?',
        r'answer\s*[is:]+\s*["\']?([^"\']+)["\']?',
    ]
    
    for pattern in quote_patterns:
        match = re.search(pattern, output_text, re.IGNORECASE)
        if match:
            extracted = match.group(1).strip()
            if candidates:
                for candidate in candidates:
                    if candidate.lower() in extracted.lower() or extracted.lower() in candidate.lower():
                        return candidate
            return extracted
    
    # ========== 默认返回第一个 candidate ==========
    return candidates[0] if candidates else "A"


def normalize_answer_to_letter(answer_text, candidates):
    """
    将答案统一转换为字母 (A, B, C, D)
    """
    if not candidates:
        return answer_text.strip().upper()
    
    answer_lower = answer_text.lower().strip()
    
    # 1. 如果本身就是字母
    if answer_lower in ['a', 'b', 'c', 'd']:
        return answer_lower.upper()
    
    # 2. 如果匹配到了候选项内容，返回对应的字母
    for i, cand in enumerate(candidates):
        if cand.lower().strip() == answer_lower or cand.lower().strip() in answer_lower:
            return chr(65 + i)
            
    # 3. 如果都不匹配，返回原始文本（作为最后手段）
    return answer_text

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
    device: str = "cuda:0",
    video_base_path: str = "/data1/lyc/datasets/MLVU/MLVU/video/2_needle"
) -> Tuple[str, str, str]:
    """
    处理单个样本（适配 MLVU 格式）
    返回：(output_text, extracted_answer, prompt_text)
    """
    video_filename = sample['video']
    # 构建完整的视频路径
    video_path = os.path.join(video_base_path, video_filename)
    
    question = sample['question']
    candidates = sample.get('candidates', [])
    ground_truth = sample.get('answer', '')
    
    # 构建问题文本（包含选项）
    options_text = "\n".join([f"{chr(65+i)}. {cand}" for i, cand in enumerate(candidates)])
    pre_prompt = """Carefully watch the video and pay attention to the cause and sequence of events, \
the detail and movement of objects, and the action and pose of persons. \
Based on your observations, you need to provide a video summary, explain your reasoning, \
and finally select the best option. \
please provide the output in the following format:
1. Video Summary: Describe what happens in the video.
2. Reasoning: Analyze why the option is correct.
3. Best Option: The final choice, strictly in the format like (A).
"""
    post_prompt = '\nPlease answer strictly following the "1. Video Summary 2. Reasoning 3. Best Option" format.\nResponse:\n1. Video Summary:'
    if OPEN_COT:
        full_question = pre_prompt + f'{question}\n{options_text}\n' + post_prompt
    else:
        full_question = f"{question}\n{options_text}\nAnswer with the option's letter from the given choices directly."
    
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
                    "nframes": NFRAMES,
                },
                {"type": "text", "text": full_question},
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
    if OPEN_COT:
        extracted_answer = extract_answer_cot(output_text, candidates, ground_truth)
    else:
        extracted_answer = extract_answer(output_text, candidates, ground_truth)
    print(f"### Extracted answer: {extracted_answer}")
    print(f"### Ground truth: {ground_truth}")

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

    args.model_type = 'qwen3vl-8b'
    args.do_sparse = DO_SPARSE

    # 初始化处理器
    print(f"Loading processor from {model_path}... ")
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
    print(f"Loading model from {args.path}... ")
    model = Qwen3VLFlexLM(args.model_type, env, args.path, policy)

    # 加载数据集
    print(f"Loading dataset from {dataset_json_path}... ")
    dataset = load_dataset(dataset_json_path)

    # 清空输出文件（如需增量添加可注释此行）
    clear_output_file(output_jsonl_path)

    # 推理循环
    print(f"Starting inference on {len(dataset)} samples... ")
    print(f"Output will be saved to {output_jsonl_path} ")
    print("=" * 60)

    success_count = 0
    error_count = 0

    # MLVU 视频基础路径
    video_base_path = "/data1/lyc/datasets/MLVU/MLVU/video/2_needle"

    ##### 只运行至少10min长视频 #####
    filtered_dataset = []
    for idx, sample in enumerate(dataset):
        if sample['duration'] >= 600: 
            filtered_dataset.append((idx, sample))

    pbar = tqdm(total=len(filtered_dataset), desc="Processing MLVU samples", unit="sample")
    for idx, sample in filtered_dataset:
        
        try:
            # 生成问题 ID
            question_id = generate_question_id(sample, idx)
            
            # 处理样本
            output_text, extracted_answer, prompt_text = process_sample( 
                sample, model, processor, args, device, video_base_path
            )
            
            # 获取 ground truth
            ground_truth = sample.get('answer', '')
            question_type = sample.get('question_type', 'unknown')

            # 统一将预测和 GT 都转换为字母再比较
            pred_letter = normalize_answer_to_letter(extracted_answer, sample['candidates'])
            gt_letter = normalize_answer_to_letter(ground_truth, sample['candidates'])
            
            # 判断是否正确（匹配答案内容）
            is_correct = pred_letter == gt_letter
            print(f"### Question ID: {question_id} | Correct: {is_correct}")
            
            # 构建结果
            result = {
                "question_id": question_id,
                "pred": extracted_answer,
                "gt": ground_truth,
                "question_type": question_type,
                "is_correct": is_correct,
                "prompt": prompt_text,
                "output": output_text,
                "question": sample['question'],
                "video": sample['video'],
                "duration": sample.get('duration', 0),
                "candidates": sample.get('candidates', [])
            }
            
            # 保存结果
            save_result(result, output_jsonl_path)
            
            success_count += 1
            
            # 每 10 个样本打印一次进度和内存信息
            if (idx + 1) % 10 == 0:
                peak_mem = torch.cuda.max_memory_allocated(device)
                print(f"\n[Progress] {idx + 1}/{len(dataset)} samples processed ")
                print(f"[Memory] CUDA peak: {peak_mem / 1024 / 1024 / 1024:.4f} GB ")
                print(f"[Success/Error] {success_count}/{error_count} ")
                print("=" * 60)
                
        except Exception as e:
            error_count += 1
            print(f"\n[ERROR] Sample {idx} failed: {str(e)} ")
            
            # 保存错误信息
            error_result = {
                "question_id": generate_question_id(sample, idx),
                "pred": "ERROR",
                "gt": sample.get('answer', ''),
                "question_type": sample.get('question_type', 'unknown'),
                "is_correct": False,
                "prompt": "",
                "output": f"ERROR: {str(e)}",
                "question": sample.get('question', ''),
                "video": sample.get('video', ''),
                "duration": sample.get('duration', 0),
                "candidates": sample.get('candidates', [])
            }
            save_result(error_result, output_jsonl_path)
            
        pbar.update(1) 

    pbar.close()

    # 最终统计
    print("\n" + "=" * 60)
    print("INFERENCE COMPLETED ")
    print("=" * 60)
    print(f"Total samples: {len(dataset)} ")
    print(f"Success: {success_count} ")
    print(f"Errors: {error_count} ")
    print(f"Output file: {output_jsonl_path} ")

    # 最终内存信息
    peak_mem = torch.cuda.max_memory_allocated(device)
    print(f"CUDA peak memory: {peak_mem / 1024 / 1024 / 1024:.4f} GB ")

if __name__ == "__main__":
    main()