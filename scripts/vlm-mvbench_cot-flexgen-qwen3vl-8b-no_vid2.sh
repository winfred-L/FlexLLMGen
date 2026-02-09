export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="/data1/lyc/hf_home"
export LMUData="/data1/lyc/LMUData" #需要手动创建

export CUDA_VISIBLE_DEVICES=0

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJ_ROOT=$(dirname "$SCRIPT_DIR")

# flexllmgen/models/qwen3vl.py 中修改index_ranges
# flexllmgen/utils.py 中修改VideoInfo.total_len
python -m VLMEvalKit.run \
    --config $SCRIPT_DIR/vlm-mvbench_cot-flexgen-qwen3vl-8b-no_vid2.json \
    --work-dir /data1/lyc/flexllmgen_outputs