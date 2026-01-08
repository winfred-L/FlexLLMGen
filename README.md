# Title

## Installation

```bash
conda create -n flexgen python=3.12 -y
conda activate flexgen
cd FlexLLMGen

git submodule update --init --recursive

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -e .
MAX_JOBS=4 pip install flash-attn --no-build-isolation

cd third_party/VLMEvalKit
pip install -e .

```