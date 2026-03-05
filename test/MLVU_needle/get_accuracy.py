import json

# json_path = '/data1/lyc/flexllmgen_outputs/MLVU_needle/MLVU_needle_over10min-qwen3vl-8b.jsonl'
# json_path = '/data1/lyc/flexllmgen_outputs/MLVU_needle/MLVU_needle_over10min-qwen3vl-8b-no_vid.jsonl'

# json_path = '/data1/lyc/flexllmgen_outputs/MLVU_needle/MLVU_needle_over10min_cot-qwen3vl-8b.jsonl'
json_path = '/data1/lyc/flexllmgen_outputs/MLVU_needle/MLVU_needle_over10min_cot-qwen3vl-8b-no_vid.jsonl'

with open(json_path, 'r', encoding='utf-8') as f:
    data = f.readlines()

total_samples = len(data)
correct_predictions = sum(1 for sample in data if json.loads(sample).get('is_correct', False))
accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0

print(f'json_path: {json_path}')
print(f'Accuracy: {accuracy:.4f} ({correct_predictions}/{total_samples})')