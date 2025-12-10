import torch
import numpy as np
import os
import glob
from tqdm import tqdm
import shutil

from safetensors.torch import load_file

# 配置你的路径
# MODEL_NAME = "opt-1.3b"
# MODEL_NAME = "qwen25vl-7b"
MODEL_NAME = "qwen3vl-8b"

# 你下载的原始模型路径 (ModelScope 下载路径)
# SOURCE_PATH = "/data/lyc/models/opt-1.3b"
SOURCE_PATH = "/data/lyc/models/Qwen2.5-VL-7B-Instruct"
# SOURCE_PATH = "/data/lyc/models/Qwen3-VL-8B-Instruct"

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



def convert_weights_qwen25vl():
    '''
    功能：
    1. 类似于 convert_weights_opt，将decoder部分权重以numpy格式保存至{MODEL_NAME}-np文件夹
    2. 将decoder_embed和encoder部分权重整理为torch的state_dict格式，保存至{MODEL_NAME}-np文件夹
    '''

    # 目标文件夹必须是 {model_name}-np 格式
    target_path = os.path.join(OUTPUT_PARENT_DIR, f"{MODEL_NAME}-np")
    os.makedirs(target_path, exist_ok=True)

    # 查找所有的 safetensors 文件
    weight_files = glob.glob(os.path.join(SOURCE_PATH, "*.safetensors"))
    if not weight_files:
        print("错误：源路径下未找到 .safetensors 文件，请检查路径。")
        return
    
    # 初始化字典用于收集特定模块的权重
    text_embed_state_dict = {}
    visual_encoder_state_dict = {}

    for weight_file in tqdm(weight_files, desc="Converting files"):
        state = load_file(weight_file, device="cpu")
        
        for name, param in tqdm(state.items(), leave=False):

            # 1. 处理 Visual Encoder 权重
            if name.startswith("visual."):
                # 去除 "visual." 前缀，以便直接加载到 VisionTransformer 模型中
                # 例如: visual.patch_embed.proj.weight -> patch_embed.proj.weight
                new_key = name.replace("visual.", "", 1)
                visual_encoder_state_dict[new_key] = param
                continue

            # 2. 处理 Text Embedding Layer 权重
            if name == "model.embed_tokens.weight":
                text_embed_state_dict["weight"] = param.clone()

            # 3. 原有的 NumPy 转换逻辑
            name = name.replace("model.", "language_model.")
            
            param_path = os.path.join(target_path, name)
            
            # 保存为 .pt 格式
            os.path.dirname(param_path) and os.makedirs(os.path.dirname(param_path), exist_ok=True)
            with open(param_path, "wb") as f:
                np.save(f, param.cpu().detach().view(torch.uint16).numpy()) # 注：numpy 不支持使用 bfloat16 格式保存，因此这里映射为 uint16

    # 保存 text_embed_layer.bin
    if text_embed_state_dict:
        torch.save(text_embed_state_dict, os.path.join(target_path, 'text_embed_layer.bin'))
    else:
        print("警告：未找到 text_embed_layer (model.embed_tokens.weight) 权重。")

    # 保存 visual_encoder.bin
    if visual_encoder_state_dict:
        torch.save(visual_encoder_state_dict, os.path.join(target_path, 'visual_encoder.bin'))
    else:
        print("警告：未找到 visual_encoder 权重。")
    
    print("转换完成！")


def convert_weights_qwen3vl():
    '''
    逻辑与 convert_weights_qwen25vl() 相同，仅对字符串匹配作修改。
    
    qwen25vl的参数名与print(model)格式前缀不同，具体区别如下：
    print(model)                state.items.name
    'model.visual.*'            'visual.*'
    'model.language_model.*'    'model.*'
    
    qwen3vl的参数名与print(model)格式完全一致
    '''

    # 目标文件夹必须是 {model_name}-np 格式
    target_path = os.path.join(OUTPUT_PARENT_DIR, f"{MODEL_NAME}-np")
    os.makedirs(target_path, exist_ok=True)

    # 查找所有的 safetensors 文件
    weight_files = glob.glob(os.path.join(SOURCE_PATH, "*.safetensors"))
    if not weight_files:
        print("错误：源路径下未找到 .safetensors 文件，请检查路径。")
        return
    
    # 初始化字典用于收集特定模块的权重
    text_embed_state_dict = {}
    visual_encoder_state_dict = {}

    for weight_file in tqdm(weight_files, desc="Converting files"):
        state = load_file(weight_file, device="cpu")
        
        for name, param in tqdm(state.items(), leave=False):

            name = name.replace("model.", "", 1)

            if name.startswith("visual."):
                new_key = name.replace("visual.", "", 1)
                visual_encoder_state_dict[new_key] = param
                continue

            if name == "language_model.embed_tokens.weight":
                text_embed_state_dict["weight"] = param.clone()
            
            param_path = os.path.join(target_path, name)
            
            # 保存为 .pt 格式
            os.path.dirname(param_path) and os.makedirs(os.path.dirname(param_path), exist_ok=True)
            with open(param_path, "wb") as f:
                np.save(f, param.cpu().detach().view(torch.uint16).numpy()) # 注：numpy 不支持使用 bfloat16 格式保存，因此这里映射为 uint16

    # 保存 text_embed_layer.bin
    if text_embed_state_dict:
        torch.save(text_embed_state_dict, os.path.join(target_path, 'text_embed_layer.bin'))
    else:
        print("警告：未找到 text_embed_layer (model.embed_tokens.weight) 权重。")

    # 保存 visual_encoder.bin
    if visual_encoder_state_dict:
        torch.save(visual_encoder_state_dict, os.path.join(target_path, 'visual_encoder.bin'))
    else:
        print("警告：未找到 visual_encoder 权重。")
    
    print("转换完成！")


if __name__ == "__main__":
    if MODEL_NAME == "opt-1.3b":
        convert_weights_opt()
    elif MODEL_NAME == "qwen25vl-7b":
        convert_weights_qwen25vl()
    elif MODEL_NAME == "qwen3vl-8b":
        convert_weights_qwen3vl()
    else:
        raise NotImplementedError(f"Model {MODEL_NAME} not supported yet.")