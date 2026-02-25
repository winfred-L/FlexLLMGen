import os
import re
from pathlib import Path

import yaml

hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
base_cache_dir = os.path.expanduser(hf_home)
with open(Path(__file__).parent / "lvbench.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)
cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]


def lvbench_doc_to_visual(doc):
    cache_dir = os.path.join(base_cache_dir, cache_name)
    video_path = doc["video_path"]
    assert os.path.exists(os.path.join(cache_dir, video_path))
    video_path = os.path.join(cache_dir, video_path)
    return [video_path]


# def lvbench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
#     print(f'{lmms_eval_specific_kwargs=}')
#     if lmms_eval_specific_kwargs is None:
#         lmms_eval_specific_kwargs = {}
#     if "pre_prompt" not in lmms_eval_specific_kwargs:
#         lmms_eval_specific_kwargs["pre_prompt"] = ""
#     if "post_prompt" not in lmms_eval_specific_kwargs:
#         lmms_eval_specific_kwargs["post_prompt"] = "\nAnswer the question with the option letter"
#     return lmms_eval_specific_kwargs["pre_prompt"] + doc["question"] + lmms_eval_specific_kwargs["post_prompt"]


# COT version
def lvbench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
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
    return pre_prompt + doc["question"] + post_prompt


# def extract_characters_regex(s):
#     s = s.strip()
#     answer_prefixes = [
#         "The best answer is",
#         "The correct answer is",
#         "The answer is",
#         "The answer",
#         "The best option is",
#         "The correct option is",
#         "Best answer:",
#         "Best option:",
#     ]
#     for answer_prefix in answer_prefixes:
#         s = s.replace(answer_prefix, "")

#     if len(s.split()) > 10 and not re.search("[ABCD]", s):
#         return ""

#     matches = re.search(r"[ABCD]", s)
#     if matches is None:
#         return ""
#     return matches[0]

# COT version
def extract_characters_regex(s):
    # 直接定位 "Best Option" 后面的 A, B, C 或 D
    # \s* 匹配可能的空格，\(? 和 \)? 匹配可能存在的括号
    match = re.search(r"3\.\s*Best Option:\s*\(?([ABCD])\)?", s, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # 作为备用方案（Fallback），如果没找到标准格式，就找文本最后出现的 (A), (B), (C) 或 (D)
    fallback_match = re.findall(r"\(([ABCD])\)", s)
    if fallback_match:
        return fallback_match[-1].upper()
        
    return ""

def lvbench_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case videomme score), value: metric value
    """
    pred = results[0]
    pred_ans = extract_characters_regex(pred)
    # gt_ans = doc["answer"].lower().strip().replace(".", "")
    gt_ans = doc["answer"]
    score = pred_ans == gt_ans

    # return {f"videomme_perception_score": data_dict for metric in matrices}
    return {f"lvbench_score": score}
