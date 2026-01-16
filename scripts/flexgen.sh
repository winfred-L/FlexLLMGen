# option: qwen25vl-7b, qwen3vl-8b
model_type="qwen3vl-8b"

# TODO: 模型权重加载不使用np，而是直接加载safetensor

python -m flexllmgen.main \
    --model-type ${model_type}