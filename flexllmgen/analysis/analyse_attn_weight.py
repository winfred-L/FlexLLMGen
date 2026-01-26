import torch
import flexllmgen.utils as utils

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
    attention_tensor (torch.Tensor): 一个一维的注意力权重张量 (通常总和为 1)。
    epsilon (float): 用于数值稳定性的极小常数，防止 log(0)。
    
    返回:
    float: 集中度分数 C，范围在 0 到 1 之间。
    """
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
shape = (max_step-1, num_layer, num_head)
sum_result = torch.zeros(shape, dtype=float)
con_result = torch.zeros(shape, dtype=float)

for step in range(1, max_step):
    for layer in range(num_layer):
        attn_weight = utils.total_attn_weight.get(layer, step)
        
        for head in range(num_head):
            vision_attn_weight = torch.cat([attn_weight[head][s : e+1] for s, e in index_ranges])

            # count all vision attn weight
            attn_weight_sum = torch.sum(vision_attn_weight)
            sum_result[step-1, layer, head] = attn_weight_sum

            # get concentration score
            concentration = calculate_concentration(vision_attn_weight)
            con_result[step-1, layer, head] = concentration


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
    plt.savefig(f'{output_path}/{title}.png', dpi=150, bbox_inches='tight')
    plt.close()

    # torch.save(matrix, f'{output_path}/{title}.pt')


output_path = '/data1/lyc/flexllmgen_outputs/attn_weight'

# sum_result_max, _ = torch.max(sum_result, dim=0)
# draw_save_layer_head(
#     matrix = sum_result_max,
#     title = 'sum_result_max',
#     output_path = output_path,
# )

# con_result_max, _ = torch.max(con_result, dim=0)
# draw_save_layer_head(
#     matrix = con_result_max,
#     title = 'con_result_max',
#     output_path = output_path,
# )
# con_result_min, _ = torch.min(con_result, dim=0)
# draw_save_layer_head(
#     matrix = con_result_min,
#     title = 'con_result_min',
#     output_path = output_path,
# )
# con_result_avg = torch.mean(con_result, dim=0)
# draw_save_layer_head(
#     matrix = con_result_avg,
#     title = 'con_result_avg',
#     output_path = output_path,
# )


general_result = sum_result * con_result
general_result_max, _ = torch.max(general_result, dim=0)
draw_save_layer_head(
    matrix = general_result_max,
    title = 'general_result_max',
    output_path = output_path,
)
general_result_min, _ = torch.min(general_result, dim=0)
draw_save_layer_head(
    matrix = general_result_min,
    title = 'general_result_min',
    output_path = output_path,
)
general_result_avg = torch.mean(general_result, dim=0)
draw_save_layer_head(
    matrix = general_result_avg,
    title = 'general_result_avg',
    output_path = output_path,
)