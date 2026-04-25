#!/usr/bin/env bash
# Submit from the repository root: sbatch hpc/train_h100.slurm

#SBATCH --job-name=pr-review-grpo
#SBATCH --partition=gpu_h100_4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH --output=logs/grpo_%j.out
#SBATCH --error=logs/grpo_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
ENV_DIR="${ENV_DIR:-/scratch/$USER/envs/pr-review}"
HF_CACHE="${HF_CACHE:-/scratch/$USER/hf_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/$USER/pr-review/grpo_checkpoint}"
REWARDS_DIR="${REWARDS_DIR:-/scratch/$USER/pr-review/rewards}"
CONDA_MODULE="${CONDA_MODULE:-anaconda3-2022.05-gcc-11.2.0-od5lltp}"

mkdir -p "$REPO_DIR/logs" "$OUTPUT_DIR" "$REWARDS_DIR"
cd "$REPO_DIR"

echo "job_id=${SLURM_JOB_ID:-local}"
echo "start=$(date)"
echo "node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv || true

module load "$CONDA_MODULE"
eval "$(conda shell.bash hook)"
conda activate "$ENV_DIR"

export HF_HOME="$HF_CACHE"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export PR_REVIEW_TOOL_BACKEND=heuristic

python - <<'PY'
import torch
from trl import GRPOConfig
print(f"python ready; cuda={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY

python tasks/build_task_bank.py --print-summary

python train/grpo_train.py \
    --preset h100 \
    --task-bank all \
    --task-loader-mode short \
    --output-dir "$OUTPUT_DIR" \
    --report-to none

python benchmarks/evaluate_baselines.py \
    --task-bank all \
    --task-loader-mode short \
    --output "$REWARDS_DIR/baseline_eval.json"

python benchmarks/evaluate_trained_model.py \
    --checkpoint "$OUTPUT_DIR" \
    --preset h100 \
    --task-bank all \
    --task-loader-mode short \
    --output "$REWARDS_DIR/trained_eval.json"

python benchmarks/generate_report.py \
    --baseline "$REWARDS_DIR/baseline_eval.json" \
    --trained "$REWARDS_DIR/trained_eval.json" \
    --log "$OUTPUT_DIR/training_log.csv" \
    --output "$REWARDS_DIR/comparison_report.png" || true

echo "finished=$(date)"
echo "checkpoint=$OUTPUT_DIR"
echo "trained_eval=$REWARDS_DIR/trained_eval.json"
