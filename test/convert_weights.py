import torch
import numpy as np
import os
import glob
from tqdm import tqdm
import shutil

from safetensors.torch import load_file

# # 配置你的路径
# MODEL_NAME = "opt-1.3b"
# # 你下载的原始模型路径 (ModelScope 下载路径)
# SOURCE_PATH = "/data/lyc/models/opt-1.3b"
# # 目标输出路径 (FlexLLMGen 读取的父目录)
# OUTPUT_PARENT_DIR = "/data/lyc/models"

# 配置你的路径
MODEL_NAME = "qwen25vl-7b"
# 你下载的原始模型路径 (ModelScope 下载路径)
SOURCE_PATH = "/data/lyc/models/Qwen2.5-VL-7B-Instruct"
# 目标输出路径 (FlexLLMGen 读取的父目录)
OUTPUT_PARENT_DIR = "/data/lyc/models"


def convert_weights_opt():
    # 目标文件夹必须是 {model_name}-np 格式
    target_path = os.path.join(OUTPUT_PARENT_DIR, f"{MODEL_NAME}-np")
    os.makedirs(target_path, exist_ok=True)
    
    print(f"正在将权重从 {SOURCE_PATH} 转换为 NumPy 格式至 {target_path} ...")

    # 查找所有的 bin 文件
    bin_files = glob.glob(os.path.join(SOURCE_PATH, "*.bin"))
    if not bin_files:
        print("错误：源路径下未找到 .bin 文件，请检查路径。")
        return

    for bin_file in tqdm(bin_files, desc="Converting files"):
        state = torch.load(bin_file, map_location="cpu")
        for name, param in tqdm(state.items(), leave=False):
            # 这里的命名替换逻辑来自 opt_config.py [cite: 166]
            name = name.replace("model.", "")
            name = name.replace("decoder.final_layer_norm", "decoder.layer_norm")
            
            param_path = os.path.join(target_path, name)
            
            # 确保父文件夹存在
            os.path.dirname(param_path) and os.makedirs(os.path.dirname(param_path), exist_ok=True)

            with open(param_path, "wb") as f:
                np.save(f, param.cpu().detach().numpy())

            # 处理共享 Embedding [cite: 167]
            if "decoder.embed_tokens.weight" in name:
                shutil.copy(param_path, param_path.replace(
                    "decoder.embed_tokens.weight", "lm_head.weight"))

    print("转换完成！")


# 注：numpy 不支持使用 bfloat16 格式保存，因此这里映射为 uint16
def convert_weights_qwen25vl():
    # 目标文件夹必须是 {model_name}-np 格式
    target_path = os.path.join(OUTPUT_PARENT_DIR, f"{MODEL_NAME}-np")
    os.makedirs(target_path, exist_ok=True)

    print(f"正在将权重从 {SOURCE_PATH} 转换为 NumPy 格式至 {target_path} ...")

    # 查找所有的 safetensors 文件
    weight_files = glob.glob(os.path.join(SOURCE_PATH, "*.safetensors"))
    if not weight_files:
        print("错误：源路径下未找到 .safetensors 文件，请检查路径。")
        return

    for weight_file in tqdm(weight_files, desc="Converting files"):
        state = load_file(weight_file, device="cpu")
        
        for name, param in tqdm(state.items(), leave=False):
            if name.startswith("visual."):
                continue

            name = name.replace("model.", "language_model.")
            
            param_path = os.path.join(target_path, name)
            
            # 保存为 .pt 格式
            os.path.dirname(param_path) and os.makedirs(os.path.dirname(param_path), exist_ok=True)
            with open(param_path, "wb") as f:
                np.save(f, param.cpu().detach().view(torch.uint16).numpy())

    print("转换完成！")


# # 注：numpy 不支持使用 bfloat16 格式保存，因此这里使用 torch 保存为 .pt 格式
# def convert_weights_qwen25vl():
#     # 目标文件夹必须是 {model_name}-pt 格式
#     target_path = os.path.join(OUTPUT_PARENT_DIR, f"{MODEL_NAME}-pt")
#     os.makedirs(target_path, exist_ok=True)

#     print(f"正在将权重从 {SOURCE_PATH} 转换为 PyTorch 格式至 {target_path} ...")

#     # 查找所有的 safetensors 文件
#     weight_files = glob.glob(os.path.join(SOURCE_PATH, "*.safetensors"))
#     if not weight_files:
#         print("错误：源路径下未找到 .safetensors 文件，请检查路径。")
#         return

#     for weight_file in tqdm(weight_files, desc="Converting files"):
#         state = load_file(weight_file, device="cpu")
        
#         for name, param in tqdm(state.items(), leave=False):
#             if name.startswith("visual."):
#                 continue

#             name = name.replace("model.", "language_model.")
            
#             param_path_base = os.path.join(target_path, name)
#             param_path = param_path_base + ".pt"
            
#             # 保存为 .pt 格式
#             os.path.dirname(param_path) and os.makedirs(os.path.dirname(param_path), exist_ok=True)
#             torch.save(param.cpu().detach(), param_path)

#     print("转换完成！")



if __name__ == "__main__":
    if MODEL_NAME == "opt-1.3b":
        convert_weights_opt()
    elif MODEL_NAME == "qwen25vl-7b":
        convert_weights_qwen25vl()