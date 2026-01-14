export CUDA_VISIBLE_DEVICES=1

# option: qwen25vl-7b, qwen3vl-8b
model_type="qwen3vl-8b"

python -m flexllmgen.main \
    --model-type ${model_type}