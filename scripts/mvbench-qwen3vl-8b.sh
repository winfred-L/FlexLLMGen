export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="/data1/lyc/hf_home"

export CUDA_VISIBLE_DEVICES=0


accelerate launch --num_processes 1 --main_process_port 12345 -m lmms_eval \
    --model qwen3_vl \
    --model_args pretrained=/data/lyc/models/Qwen3-VL-8B-Instruct,max_pixels=786432,attn_implementation=flash_attention_2,max_num_frames=64 \
    --tasks mvbench \
    --batch_size 1 \
    --log_samples \
    --output_path /data1/lyc/flexllmgen_outputs/logs/