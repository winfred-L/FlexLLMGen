# FlexCacheGen

## Installation

Step 1.

```bash
conda create -n flexgen python=3.12 -y
conda activate flexgen
cd FlexLLMGen

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
MAX_JOBS=4 pip install flash-attn --no-build-isolation

cd lmms-eval
pip install -e .

cd VLMEvalKit
pip install -e .

cd ..
pip install -e .
```

Step 2.

set up `.env`.


## download

```bash
# set huggingface token
export HF_HOME="/data1/lyc/hf_home"
git config --global credential.helper store
hf auth login

# optional: manually download datasets
export HF_ENDPOINT="https://hf-mirror.com"
hf download Qwen/Qwen2.5-7B-Instruct
hf download Qwen/Qwen3-VL-8B-Instruct
hf download lmms-lab/VideoChatGPT --repo-type dataset
hf download OpenGVLab/MVBench --repo-type dataset
```


## modification in lmms-eval and VLMEvalKit

add `lmms-eval/lmms_eval/models/simple/flexgenlm.py`
modify `lmms-eval/lmms_eval/models/__init__.py` with a new line, to add `flexgenlm`

add `VLMEvalKit/vlmeval/vlm/flexgenlm.py`
modify `VLMEvalKit/vlmeval/vlm/__init__.py` with a new line, to add `flexgenlm`