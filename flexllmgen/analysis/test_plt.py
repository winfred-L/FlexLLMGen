import matplotlib.pyplot as plt
import torch
import numpy as np




def draw_2d_scatter(
    sum_data: torch.Tensor, # shape: (step)
    con_data: torch.Tensor, # shape: (step)
    title: str,
    output_path: str
):
    x_data = sum_data.detach().cpu().numpy().flatten()
    y_data = con_data.detach().cpu().numpy().flatten()

    # 设置画布大小和清晰度
    plt.figure(figsize=(6, 6), dpi=100)

    # 绘制散点图
    # s=1.0: 点的大小，数据量大时要设置得很小
    # alpha=0.3: 透明度，这是关键！透明度能让你看到点的堆叠密度（颜色深的地方说明点多）
    plt.scatter(x_data, y_data, s=10.0, c='tab:blue', alpha=0.5, edgecolors='none')

    # 设置轴标签
    plt.xlabel('sum', fontsize=12)
    plt.ylabel('concentration', fontsize=12)

    # 设置标题 (可根据你的模型名称修改)
    plt.title(title, fontsize=14)

    # 设置坐标轴范围 (根据你的描述是 0-1)
    # 稍微留一点边距 (-0.05 到 1.05) 会更好看，或者严格限制在 (0, 1)
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    # 调整布局
    plt.tight_layout()

    # 保存图片
    plt.savefig(f'{output_path}/{title}.png', dpi=150, bbox_inches='tight')
    plt.close()



# 模拟数据
# 假设维度：Layer=32, Head=32, Step=100
shape = (32, 32, 100)
# 随机生成 0-1 之间的数据来模拟你的 tensor
sum_result = torch.rand(shape) 
# 为了让图看起来像样例，稍微加点相关性
con_result = torch.clamp(torch.abs(torch.randn(shape) * 0.5 + 0.3), 0, 1) 


layer = 0
head = 0
output_path = '/data1/lyc/flexllmgen_outputs/attn_weight'
draw_2d_scatter(
    sum_data=sum_result[layer, head, :],
    con_data=con_result[layer, head, :],
    title=f'layer{layer}_head{head}',
    output_path=output_path
)