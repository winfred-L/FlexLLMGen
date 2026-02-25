import re
import json

# 原始的、存在缺陷的提取函数
def extract_characters_regex_old(s):
    s = s.strip()
    answer_prefixes = [
        "The best answer is", "The correct answer is", "The answer is",
        "The answer", "The best option is", "The correct option is",
        "Best answer:", "Best option:",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    if len(s.split()) > 10 and not re.search("[ABCD]", s):
        return ""

    matches = re.search(r"[ABCD]", s)
    if matches is None:
        return ""
    return matches[0]

# 改进后的提取函数（根据你严格的输出格式 "3. Best Option: (X)" 设计）
def extract_characters_regex_new(s):
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

def evaluate_accuracy(jsonl_file_path):
    total = 0
    old_correct = 0
    new_correct = 0
    mismatches = []

    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            
            data = json.loads(line)
            target = data.get("target", "").strip().upper()
            
            # 假设只评估 filtered_resps 中的第一个回答
            if not data.get("filtered_resps"):
                continue
                
            response_text = data["filtered_resps"][0]
            
            # 分别使用新旧逻辑提取
            old_pred = extract_characters_regex_old(response_text)
            new_pred = extract_characters_regex_new(response_text)
            
            total += 1
            if old_pred == target:
                old_correct += 1
            if new_pred == target:
                new_correct += 1
                
            # 记录提取结果不一致的样本，方便手动排查
            if old_pred != new_pred:
                mismatches.append({
                    "doc_id": data.get("doc_id", line_num),
                    "target": target,
                    "old_extracted": old_pred,
                    "new_extracted": new_pred,
                    "response_snippet": response_text[-100:] # 只看最后100个字符用于debug
                })

    print(f"--- 评测结果统计 ---")
    print(f"总样本数: {total}")
    print(f"旧逻辑 (存在缺陷) 准确率: {old_correct / total * 100:.2f}% ({old_correct}/{total})")
    print(f"新逻辑 (修复后) 准确率:   {new_correct / total * 100:.2f}% ({new_correct}/{total})")
    print(f"\n发现 {len(mismatches)} 个提取结果不一致的样本。")
    
    # # 打印前 5 个不一致的样本供检查
    # for m in mismatches[:5]:
    #     print(f"Doc ID: {m['doc_id']} | 真实答案: {m['target']} | 旧提取: {m['old_extracted']} | 新提取: {m['new_extracted']}")



if __name__ == "__main__":
    # jsonl_path = '/data1/lyc/flexllmgen_outputs/lmm-lvbench-flexgen-qwen3vl-8b/20260219_214412_samples_lvbench.jsonl'
    jsonl_path = '/data1/lyc/flexllmgen_outputs/lmm-lvbench-flexgen-qwen3vl-8b-no_vid/20260220_132740_samples_lvbench.jsonl'

    evaluate_accuracy(jsonl_path) 
    