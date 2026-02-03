export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="/data1/lyc/hf_home"
export LMUData="/data1/lyc/LMUData" #需要手动创建

export CUDA_VISIBLE_DEVICES=1

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJ_ROOT=$(dirname "$SCRIPT_DIR")

# eager analyse attn weight
python -m VLMEvalKit.run \
    --config $SCRIPT_DIR/vlm-mvbench_cot-flexgentest-qwen3vl-8b.json \
    --work-dir /data1/lyc/flexllmgen_outputs \
    --mode infer