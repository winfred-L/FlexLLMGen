import torch
import time
import random

from transformers import Qwen3VLForConditionalGeneration

models = []
for i in range(4):
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "/data/lyc/models/Qwen3-VL-8B-Instruct",
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to("cuda:0")
    models.append(model)

while True:
    time.sleep(random.randint(1, 5) * 60) # seconds
    model = models.pop()
    del model
    
    time.sleep(random.randint(1, 5) * 60) # seconds
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "/data/lyc/models/Qwen3-VL-8B-Instruct",
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to("cuda:0")
    models.append(model)
