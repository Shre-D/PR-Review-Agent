#!/usr/bin/env bash
# Prepare a Python 3.11 training environment on an HPC login node.

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_DIR="${ENV_DIR:-/scratch/$USER/envs/pr-review}"
HF_CACHE="${HF_CACHE:-/scratch/$USER/hf_cache}"
CONDA_MODULE="${CONDA_MODULE:-anaconda3-2022.05-gcc-11.2.0-od5lltp}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

echo "Loading conda module: $CONDA_MODULE"
module load "$CONDA_MODULE"
eval "$(conda shell.bash hook)"

if [ -d "$ENV_DIR" ]; then
    echo "Using existing environment: $ENV_DIR"
    conda activate "$ENV_DIR"
else
    echo "Creating environment: $ENV_DIR"
    conda create -y -p "$ENV_DIR" "python=$PYTHON_VERSION"
    conda activate "$ENV_DIR"
fi

cd "$REPO_DIR"

python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e ".[server,train,dev]"
python -m pip install matplotlib numpy pandas

mkdir -p "$HF_CACHE"
export HF_HOME="$HF_CACHE"

python - <<'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "Qwen/Qwen3-1.7B"
print(f"Pre-downloading {model_id}")
AutoTokenizer.from_pretrained(model_id)
AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
print("Model cache ready")
PY

python - <<'PY'
import torch
print(f"torch={torch.__version__} cuda={torch.version.cuda} cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu={torch.cuda.get_device_name(0)}")
PY

echo "Setup complete"
echo "  repo:  $REPO_DIR"
echo "  env:   $ENV_DIR"
echo "  cache: $HF_CACHE"
echo "Next: sbatch hpc/train_h100.slurm"
