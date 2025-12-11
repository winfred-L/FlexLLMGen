import torch
from transformers import AutoModelForCausalLM, AutoConfig, Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration, LlavaNextVideoForConditionalGeneration


def print_opt_weights():
    model = AutoModelForCausalLM.from_pretrained(
        '/data/lyc/models/opt-1.3b', 
        trust_remote_code=True,
        device_map="cpu",
        torch_dtype=torch.float16
    )
        
    print("Model: opt-1.3b")
    print(model)
    print()
    print(f"{'Param Name':<60} | {'Shape'}")
    print("-" * 80)
    
    for name, param in model.named_parameters():
        print(f"{name:<60} | {list(param.shape)}")

def print_qwen25vl_weights():
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        '/data/lyc/models/Qwen2.5-VL-7B-Instruct',
        dtype=torch.bfloat16,
        attn_implementation='eager',
        device_map="cpu",
        trust_remote_code=True,
    )
        
    print("Model: Qwen2.5-VL-7B-Instruct")
    print(model)
    print()
    print(f"{'Param Name':<60} | {'Shape'}")
    print("-" * 80)
    
    for name, param in model.named_parameters():
        print(f"{name:<60} | {list(param.shape)}")

def print_qwen3vl_weights():
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        '/data/lyc/models/Qwen3-VL-8B-Instruct',
        dtype=torch.bfloat16,
        attn_implementation='eager',
        device_map="cpu",
        trust_remote_code=True,
    )
        
    print("Model: Qwen3-VL-8B-Instruct")
    print(model)
    print()
    print(f"{'Param Name':<60} | {'Shape'}")
    print("-" * 80)
    
    for name, param in model.named_parameters():
        print(f"{name:<60} | {list(param.shape)}")


def print_llava_vicuna_weights():
    model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        '/data/lyc/models/LLaVA-NeXT-Video-7B-hf',
        attn_implementation = 'eager',
        dtype = torch.float16, 
        low_cpu_mem_usage = True, 
    )

    print("Model: LLaVA-NeXT-Video-7B-hf")
    print(model)
    print()
    print(f"{'Param Name':<60} | {'Shape'}")
    print("-" * 80)
    
    for name, param in model.named_parameters():
        print(f"{name:<60} | {list(param.shape)}")


if __name__ == "__main__":
    # print_opt_weights()
    # print_qwen25vl_weights()
    # print_qwen3vl_weights()
    print_llava_vicuna_weights()