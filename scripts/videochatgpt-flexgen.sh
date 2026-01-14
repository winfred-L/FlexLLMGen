export EVAL_MODEL_NAME="gpt-5-mini-2025-08-07"
export MODEL_VERSION="gpt-5-mini-2025-08-07"
export OPENAI_API_URL="https://yunwu.ai/v1/chat/completions"
export OPENAI_API_KEY="sk-yb4fxa2WVsZunr0G2OHz56QPjTHxgSnYBJD6GZE6EUYtK1ZN"

export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="/data1/lyc/hf_home"

export CUDA_VISIBLE_DEVICES=0


accelerate launch --num_processes 1 --main_process_port 12345 -m lmms_eval \
    --model flexgenlm \
    --model_args model-type=qwen3vl-8b \
    --tasks videochatgpt \
    --batch_size 1 \
    --log_samples \
    --output_path /data1/lyc/flexllmgen_outputs/logs/ \
    --limit 1 \
    --verbosity DEBUG