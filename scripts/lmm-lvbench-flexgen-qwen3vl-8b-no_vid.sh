export HF_ENDPOINT="https://hf-mirror.com" # clashoff
export HF_HOME="/data1/lyc/hf_home"

export CUDA_VISIBLE_DEVICES=1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

accelerate launch --num_processes 1 --main_process_port 12345 -m lmms_eval \
    --model flexgenlm \
    --model_args model-type=qwen3vl-8b,do-sparse=True \
    --tasks lvbench \
    --batch_size 1 \
    --log_samples \
    --output_path /data1/lyc/flexllmgen_outputs/lmm-lvbench-flexgen-qwen3vl-8b-no_vid/