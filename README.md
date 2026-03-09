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


# install decord GPU version (unused?)
git clone --recursive https://github.com/dmlc/decord
cd decord
mkdir build && cd build
cmake .. -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cd ../python
python setup.py install
cd /home/lyc/miniconda3/envs/flexgen/lib/python3.12/site-packages/decord/
ln -s /home/lyc/FlexLLMGen/decord/build/libdecord.so libdecord.so


# install torchcodec
# https://github.com/meta-pytorch/torchcodec?tab=readme-ov-file#installing-torchcodec
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
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
hf download Qwen/Qwen2.5-VL-7B-Instruct
hf download Qwen/Qwen3-VL-8B-Instruct
hf download lmms-lab/VideoChatGPT --repo-type dataset
hf download OpenGVLab/MVBench --repo-type dataset
hf download lmms-lab/LVBench --repo-type dataset

modelscope download --dataset lmms-lab/Video-MME --local_dir /data1/lyc/datasets/Video-MME
modelscope download --dataset AI-ModelScope/MLVU --local_dir /data1/lyc/datasets/MLVU --include 'MLVU/video/2_needle/*'
modelscope download --dataset AI-ModelScope/MLVU --local_dir /data1/lyc/datasets/MLVU --include 'MLVU/video/9_summary/*'
```


## modification in lmms-eval and VLMEvalKit

add `lmms-eval/lmms_eval/models/simple/flexgenlm.py`
modify `lmms-eval/lmms_eval/models/__init__.py` with a new line, to add `flexgenlm`

add `VLMEvalKit/vlmeval/vlm/flexgenlm.py`
modify `VLMEvalKit/vlmeval/vlm/__init__.py` with a new line, to add `flexgenlm`

modify `VLMEvalKit/vlmeval/dataset/mvbench.py` to add class `MVBench_MP4_CoT`
modify `VLMEvalKit/vlmeval/dataset/video_dataset_config.py` to add `MVBench_MP4_CoT_1fps`
modify `VLMEvalKit/vlmeval/dataset/__init__.py` to add `MVBench_MP4_CoT`