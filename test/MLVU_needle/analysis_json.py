import json
import statistics
from collections import defaultdict
from typing import List, Dict, Any

def load_dataset(json_path: str) -> List[dict]:
    """加载 JSON 格式的数据集"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples from {json_path}")
    return data

def analyze_duration_distribution(data: List[dict], duration_key: str = 'duration') -> Dict[str, Any]:
    """
    分析 duration 字段的分布情况
    返回统计信息字典
    """
    # 提取所有 duration 值
    durations = []
    missing_count = 0
    
    for sample in data:
        duration = sample.get(duration_key)
        if duration is not None:
            try:
                durations.append(float(duration))
            except (ValueError, TypeError):
                missing_count += 1
        else:
            missing_count += 1
    
    if not durations:
        print("No valid duration data found!")
        return {}
    
    # 基础统计
    stats = {
        'total_samples': len(data),
        'valid_samples': len(durations),
        'missing_samples': missing_count,
        'min': min(durations),
        'max': max(durations),
        'mean': statistics.mean(durations),
        'median': statistics.median(durations),
        'stdev': statistics.stdev(durations) if len(durations) > 1 else 0,
    }
    
    # 分位数统计
    sorted_durations = sorted(durations)
    n = len(sorted_durations)
    stats['percentiles'] = {
        'p10': sorted_durations[int(n * 0.1)],
        'p25': sorted_durations[int(n * 0.25)],
        'p50': sorted_durations[int(n * 0.5)],
        'p75': sorted_durations[int(n * 0.75)],
        'p90': sorted_durations[int(n * 0.9)],
        'p95': sorted_durations[int(n * 0.95)],
        'p99': sorted_durations[int(n * 0.99)],
    }
    
    # 时长区间分布（按分钟分段）
    duration_ranges = [
        (0, 60, '0-1 min'),
        (60, 180, '1-3 min'),
        (180, 300, '3-5 min'),
        (300, 600, '5-10 min'),
        (600, 900, '10-15 min'),
        (900, 1800, '15-30 min'),
        (1800, float('inf'), '>30 min'),
    ]
    
    range_distribution = defaultdict(int)
    for dur in durations:
        for low, high, label in duration_ranges:
            if low <= dur < high:
                range_distribution[label] += 1
                break
    
    stats['range_distribution'] = dict(range_distribution)
    stats['range_distribution_percent'] = {
        k: f"{v / len(durations) * 100:.2f}%" 
        for k, v in range_distribution.items()
    }
    
    return stats

def print_statistics(stats: Dict[str, Any]):
    """打印统计信息"""
    print("\n" + "=" * 60)
    print("DURATION DISTRIBUTION STATISTICS")
    print("=" * 60)
    
    print(f"\n【样本数量】")
    print(f"  总样本数：    {stats.get('total_samples', 'N/A')}")
    print(f"  有效样本数：  {stats.get('valid_samples', 'N/A')}")
    print(f"  缺失样本数：  {stats.get('missing_samples', 'N/A')}")
    
    print(f"\n【基础统计（秒）】")
    print(f"  最小值：      {stats.get('min', 'N/A'):.2f} s  ({stats.get('min', 0)/60:.2f} min)")
    print(f"  最大值：      {stats.get('max', 'N/A'):.2f} s  ({stats.get('max', 0)/60:.2f} min)")
    print(f"  平均值：      {stats.get('mean', 'N/A'):.2f} s  ({stats.get('mean', 0)/60:.2f} min)")
    print(f"  中位数：      {stats.get('median', 'N/A'):.2f} s  ({stats.get('median', 0)/60:.2f} min)")
    print(f"  标准差：      {stats.get('stdev', 'N/A'):.2f} s  ({stats.get('stdev', 0)/60:.2f} min)")
    
    print(f"\n【分位数统计（秒）】")
    percentiles = stats.get('percentiles', {})
    for key, value in percentiles.items():
        print(f"  {key.upper()}:  {value:.2f} s  ({value/60:.2f} min)")
    
    print(f"\n【时长区间分布】")
    range_dist = stats.get('range_distribution', {})
    range_percent = stats.get('range_distribution_percent', {})
    for range_label in range_dist.keys():
        count = range_dist[range_label]
        percent = range_percent.get(range_label, 'N/A')
        print(f"  {range_label:12s}: {count:5d}  ({percent})")
    
    print("\n" + "=" * 60)

def plot_duration_histogram(durations: List[float], output_path: str = 'duration_distribution.png'):
    """
    绘制 duration 直方图（需要 matplotlib）
    """
    try:
        import matplotlib.pyplot as plt
        
        # 转换为分钟
        durations_min = [d / 60 for d in durations]
        
        plt.figure(figsize=(12, 6))
        plt.hist(durations_min, bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('Duration (minutes)')
        plt.ylabel('Count')
        plt.title('Video Duration Distribution')
        plt.grid(True, alpha=0.3)
        
        # 添加平均线和median线
        mean_min = statistics.mean(durations_min)
        median_min = statistics.median(durations_min)
        plt.axvline(mean_min, color='r', linestyle='--', label=f'Mean: {mean_min:.2f} min')
        plt.axvline(median_min, color='g', linestyle='--', label=f'Median: {median_min:.2f} min')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        print(f"\nHistogram saved to: {output_path}")
        
    except ImportError:
        print("matplotlib not installed. Skip plotting.")

# ==================== 主程序 ====================
if __name__ == "__main__":
    # 配置数据集路径
    dataset_json_path = "/data1/lyc/datasets/MLVU/MLVU/json/2_needle.json"
    
    # 加载数据
    data = load_dataset(dataset_json_path)
    
    # 分析分布
    stats = analyze_duration_distribution(data, duration_key='duration')
    
    # 打印统计信息
    if stats:
        print_statistics(stats)
        
        # 可选：绘制直方图
        # 提取 durations 用于绘图
        durations = [s.get('duration') for s in data if s.get('duration') is not None]
        plot_duration_histogram(durations, output_path='mlvu_needle_duration_dist.png')


'''
============================================================
DURATION DISTRIBUTION STATISTICS
============================================================

【样本数量】
  总样本数：    355
  有效样本数：  355
  缺失样本数：  0

【基础统计（秒）】
  最小值：      239.06 s  (3.98 min)
  最大值：      8332.90 s  (138.88 min)
  平均值：      858.48 s  (14.31 min)
  中位数：      471.23 s  (7.85 min)
  标准差：      1372.20 s  (22.87 min)

【分位数统计（秒）】
  P10:  455.36 s  (7.59 min)
  P25:  461.60 s  (7.69 min)
  P50:  471.23 s  (7.85 min)
  P75:  558.07 s  (9.30 min)
  P90:  867.89 s  (14.46 min)
  P95:  5487.65 s  (91.46 min)
  P99:  7421.88 s  (123.70 min)

【时长区间分布】
  5-10 min    :   266  (74.93%)
  15-30 min   :    14  (3.94%)
  10-15 min   :    50  (14.08%)
  >30 min     :    18  (5.07%)
  3-5 min     :     7  (1.97%)

============================================================
'''