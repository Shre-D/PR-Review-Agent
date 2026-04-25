#!/bin/bash
# setup_env.sh — run ONCE on the login node before submitting the training job.
#
# Usage:
#   cd ~/PR-Review-Agent
#   bash hpc/setup_env.sh
#
# What it does:
#   1. Loads CUDA 12.1 + Anaconda modules
#   2. Creates a conda env in /scratch (preserves home quota)
#   3. Installs project + ML dependencies
#   4. Pre-downloads Qwen3-1.7B weights to /scratch/hf_cache
#      (compute nodes may not have outbound internet)

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="/scratch/kulkarnis/envs/pr-review"
HF_CACHE="/scratch/kulkarnis/hf_cache"

echo "=== Loading modules ==="
module load cuda-12.1.0-gcc-11.2.0-s5o57xp
module load anaconda3-2022.05-gcc-11.2.0-od5lltp

echo "=== Creating conda env at $ENV_DIR ==="
if [ -d "$ENV_DIR" ]; then
    echo "  Already exists — skipping create"
else
    conda create -y -p "$ENV_DIR" python=3.10
fi

# shellcheck disable=SC1091
source activate "$ENV_DIR"

echo "=== Installing project ==="
cd "$REPO_DIR"
pip install -e ".[server,dev]" --quiet

echo "=== Installing ML deps (CUDA 12.1) ==="
pip install torch --index-url https://download.pytorch.org/whl/cu121 --quiet
pip install transformers trl peft bitsandbytes accelerate datasets --quiet
pip install matplotlib numpy --quiet

echo "=== Verifying GPU stack ==="
python - <<'EOF'
import torch
assert torch.cuda.is_available(), "CUDA not visible to Python — check module load"
print(f"  torch {torch.__version__}  cuda {torch.version.cuda}  devices={torch.cuda.device_count()}")
EOF

echo "=== Pre-downloading Qwen/Qwen3-1.7B to $HF_CACHE ==="
mkdir -p "$HF_CACHE"
export HF_HOME="$HF_CACHE"
python - <<'EOF'
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
print("  Downloading tokenizer...")
AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
print("  Downloading model weights (bf16, ~3.5 GB)...")
AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", torch_dtype=torch.bfloat16)
print("  Download complete.")
EOF

echo ""
echo "=== Setup complete ==="
echo "  Conda env : $ENV_DIR"
echo "  HF cache  : $HF_CACHE"
echo ""
echo "Next step:"
echo "  sbatch hpc/train_h100.slurm"
