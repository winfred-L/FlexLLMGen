import torch
import flexllmgen.utils as utils
import os

######### set up input info #########
model_type = 'qwen3vl-8b'
video_id = '28s'

# copied from print
max_step = 191
index_ranges = [(10, 414), (423, 827), (836, 1240), (1249, 1653), (1662, 2066), (2075, 2479), (2488, 2892), (2901, 3305), (3314, 3718), (3727, 4131), (4141, 4545), (4555, 4959), (4969, 5373), (5383, 5787), (5797, 6201), (6211, 6615), (6625, 7029), (7039, 7443), (7453, 7857), (7867, 8271), (8281, 8685), (8695, 9099), (9109, 9513), (9523, 9927), (9937, 10341), (10351, 10755), (10765, 11169), (11179, 11583)] # end inclusive
#####################################


utils.total_attn_weight.setup(model_type, video_id)
utils.total_attn_weight.load()

if model_type == 'qwen3vl-8b':
    num_layer = 36
    num_head = 32
    num_kv_head = 8 # unuse

def calculate_concentration(attention_tensor, epsilon=1e-9):
    """
    根据公式计算注意力向量的集中度分数 C。
    
    参数:
    attention_tensor (torch.Tensor): 一个一维的注意力权重张量 (通常总和不为 1, 函数内先进行归一化)。
    epsilon (float): 用于数值稳定性的极小常数，防止 log(0)。
    
    返回:
    float: 集中度分数 C，范围在 0 到 1 之间。
    """
    # 0. 归一化
    attn_sum = torch.sum(attention_tensor)
    if attn_sum > 0.05: # 筛选注意力权重累计和大于一定阈值的位置
        attention_tensor = attention_tensor / attn_sum
    else: # 如果总和太小，直接返回0
        return 0.0

    # 1. 获取 token 的数量 N
    N = attention_tensor.size(0)
    
    # 2. 计算熵 H = -sum(alpha * log2(alpha + epsilon))
    # 注意：为了与分母的 log2 保持底数一致，这里使用 torch.log2
    entropy = -torch.sum(attention_tensor * torch.log2(attention_tensor + epsilon))
    
    # 3. 计算最大可能熵 log2(N + epsilon)
    max_entropy = torch.log2(torch.tensor(N + epsilon, dtype=attention_tensor.dtype))
    
    # 4. 计算集中度分数 C = 1 - (H / max_entropy)
    concentration_score = 1 - (entropy / max_entropy)
    
    return concentration_score.item()


######### process attn weight #########
shape = (num_layer, num_head, max_step-1)
sum_result = torch.zeros(shape, dtype=float)
con_result = torch.zeros(shape, dtype=float)

for step in range(1, max_step):
    for layer in range(num_layer):
        attn_weight = utils.total_attn_weight.get(layer, step)
        
        for head in range(num_head):
            vision_attn_weight = torch.cat([attn_weight[head][s : e+1] for s, e in index_ranges])

            # count all vision attn weight
            attn_weight_sum = torch.sum(vision_attn_weight)
            sum_result[layer, head, step-1] = attn_weight_sum

            # get concentration score
            concentration = calculate_concentration(vision_attn_weight)
            con_result[layer, head, step-1] = concentration


######### draw graph #########
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def draw_save_layer_head(
    matrix: torch.Tensor,
    title: str,
    output_path: str
):
    vmin = 0.0
    vmax = 1.0

    colors = [(1, 1, 1), (1, 0, 0)]  # white -> red
    cmap_name = 'white_to_red'
    white_to_red_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=256)

    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap=white_to_red_cmap, aspect='auto', vmin=vmin, vmax=vmax)
    plt.colorbar(label='Value')
    plt.title(title)
    plt.xlabel('head index')
    plt.ylabel('layer index')

    # Annotate each cell with the value (formatted to 2 decimal places)
    for i in range(matrix.size(0)):  # layers
        for j in range(matrix.size(1)):  # heads
            if matrix[i, j] != 0.0:
                plt.text(j, i, f'{matrix[i, j]:.2f}',
                            ha='center', va='center',
                            color='black' if (matrix[i, j] - vmin) / (vmax - vmin + 1e-8) < 0.5 else 'white',
                            fontsize=5)
            
    plt.tight_layout()
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    plt.savefig(f'{output_path}/{title}.png', dpi=150, bbox_inches='tight')
    plt.close()

    # torch.save(matrix, f'{output_path}/{title}.pt')


output_path = '/data1/lyc/flexllmgen_outputs/attn_weight/qwen3vl-8b_28s'

sum_result_max, _ = torch.max(sum_result, dim=2)
print(f'sum_result_max: {sum_result_max}')
torch.save(sum_result_max, f'{output_path}/sum_result_max.pt')
import pdb; pdb.set_trace()
# draw_save_layer_head(
#     matrix = sum_result_max,
#     title = 'sum_result_max',
#     output_path = output_path,
# )
# sum_result_min, _ = torch.min(sum_result, dim=2)
# draw_save_layer_head(
#     matrix = sum_result_min,
#     title = 'sum_result_min',
#     output_path = output_path,
# )
# sum_result_avg = torch.mean(sum_result, dim=2)
# draw_save_layer_head(
#     matrix = sum_result_avg,
#     title = 'sum_result_avg',
#     output_path = output_path,
# )

# con_result_max, _ = torch.max(con_result, dim=2)
# draw_save_layer_head(
#     matrix = con_result_max,
#     title = 'con_result_max',
#     output_path = output_path,
# )
# con_result_min, _ = torch.min(con_result, dim=2)
# draw_save_layer_head(
#     matrix = con_result_min,
#     title = 'con_result_min',
#     output_path = output_path,
# )
# con_result_avg = torch.mean(con_result, dim=2)
# draw_save_layer_head(
#     matrix = con_result_avg,
#     title = 'con_result_avg',
#     output_path = output_path,
# )


# general_result = (sum_result + con_result) / 2   # 即投影到 y=x 主对角线
# general_result_max, _ = torch.max(general_result, dim=2)
# draw_save_layer_head(
#     matrix = general_result_max,
#     title = 'general_result_max',
#     output_path = output_path,
# )
# general_result_min, _ = torch.min(general_result, dim=2)
# draw_save_layer_head(
#     matrix = general_result_min,
#     title = 'general_result_min',
#     output_path = output_path,
# )
# general_result_avg = torch.mean(general_result, dim=2)
# draw_save_layer_head(
#     matrix = general_result_avg,
#     title = 'general_result_avg',
#     output_path = output_path,
# )


score_result = sum_result * (1 + con_result)
# 压缩范围至0-1
score_result_max_element = torch.max(score_result)
score_result = score_result / score_result_max_element

score_result_max, _ = torch.max(score_result, dim=2)
draw_save_layer_head(
    matrix = score_result_max,
    title = 'score_result_max',
    output_path = output_path,
)
score_result_min, _ = torch.min(score_result, dim=2)
draw_save_layer_head(
    matrix = score_result_min,
    title = 'score_result_min',
    output_path = output_path,
)
score_result_avg = torch.mean(score_result, dim=2)
draw_save_layer_head(
    matrix = score_result_avg,
    title = 'score_result_avg',
    output_path = output_path,
)



### 绘制各layer-head组合下 所有step的sum_result和con_result二维散点图


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
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    plt.savefig(f'{output_path}/{title}.png', dpi=150, bbox_inches='tight')
    plt.close()


# for layer in range(num_layer):
#     for head in range(num_head):
#         draw_2d_scatter(
#             sum_data=sum_result[layer, head, :],
#             con_data=con_result[layer, head, :],
#             title=f'layer{layer}_head{head}',
#             output_path=f'{output_path}/{model_type}_{video_id}'
#         )