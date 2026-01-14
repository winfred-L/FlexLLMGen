# FlexCacheGen

## Installation

```bash
conda create -n flexgen python=3.12 -y
conda activate flexgen
cd FlexLLMGen

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -e .
MAX_JOBS=4 pip install flash-attn --no-build-isolation

cd lmms-eval
pip install -e .
```


## download

```bash
# set huggingface token
export HF_HOME="/data1/lyc/hf_home"
git config --global credential.helper store
hf auth login

# optional: manually download datasets
export HF_ENDPOINT="https://hf-mirror.com"
hf download lmms-lab/VideoChatGPT --repo-type dataset
hf download OpenGVLab/MVBench --repo-type dataset
```


## modification in lmms-eval

add `lmms-eval/lmms_eval/models/simple/flexgenlm.py`
modify `lmms-eval/lmms_eval/models/__init__.py` with a new line, to add `flexgenlm`