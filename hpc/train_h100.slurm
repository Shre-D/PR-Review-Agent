#!/bin/bash
# train_h100.slurm — GRPO training job for the PR review agent on H100.
#
# Submit from the repo root:
#   sbatch hpc/train_h100.slurm
#
# Prerequisites:
#   bash hpc/setup_env.sh    (run once on login node first)
#
# NOTE: If your account requires a QOS flag, add:
#   #SBATCH --qos=<qos_name>
# Run `sacctmgr show qos | grep h100` on the login node to find the full name.

#SBATCH --job-name=pr-review-grpo
#SBATCH --partition=gpu_h100_4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=/home/kulkarnis/PR-Review-Agent/logs/grpo_%j.out
#SBATCH --error=/home/kulkarnis/PR-Review-Agent/logs/grpo_%j.err

set -euo pipefail

REPO_DIR="$HOME/PR-Review-Agent"
ENV_DIR="/scratch/kulkarnis/envs/pr-review"
HF_CACHE="/scratch/kulkarnis/hf_cache"
OUTPUT_DIR="/scratch/kulkarnis/grpo_checkpoint"
REWARDS_DIR="/scratch/kulkarnis/rewards"

# --- Modules ---
module load cuda-12.1.0-gcc-11.2.0-s5o57xp
module load anaconda3-2022.05-gcc-11.2.0-od5lltp

# --- Activate env ---
# shellcheck disable=SC1091
source activate "$ENV_DIR"

# --- Environment variables ---
export HF_HOME="$HF_CACHE"
export TRANSFORMERS_OFFLINE=1     # use pre-downloaded weights; no network needed
export HF_DATASETS_OFFLINE=1
export PR_REVIEW_TOOL_BACKEND=heuristic   # no external tool calls during training

# --- Dirs ---
mkdir -p "$OUTPUT_DIR" "$REWARDS_DIR" "$REPO_DIR/logs"

# --- Diagnostics ---
echo "========================================"
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $(hostname)"
echo "GPU      : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Python   : $(python --version)"
echo "========================================"
nvidia-smi

cd "$REPO_DIR"

# --- Train ---
echo ""
echo "=== Starting GRPO training (h100 preset) ==="
python train/grpo_train.py \
    --preset h100 \
    --output-dir "$OUTPUT_DIR" \
    --report-to none

echo ""
echo "=== Training complete. Checkpoint at $OUTPUT_DIR ==="

# --- Evaluate trained model vs baselines ---
echo ""
echo "=== Running evaluation on trained checkpoint ==="
python benchmarks/evaluate_trained_model.py \
    --checkpoint "$OUTPUT_DIR" \
    --preset h100 \
    --output "$REWARDS_DIR/trained_eval.json"

# Copy baseline artifact alongside trained eval so generate_report can find both
cp "$REPO_DIR/rewards/baseline_eval.json" "$REWARDS_DIR/baseline_eval.json"

# --- Generate visual report ---
echo ""
echo "=== Generating comparison report ==="
python benchmarks/generate_report.py \
    --baseline "$REWARDS_DIR/baseline_eval.json" \
    --trained  "$REWARDS_DIR/trained_eval.json" \
    --log      "$OUTPUT_DIR/training_log.csv" \
    --output   "$REWARDS_DIR/comparison_report.png"

echo ""
echo "========================================"
echo "Artifacts:"
echo "  Checkpoint : $OUTPUT_DIR"
echo "  Trained eval: $REWARDS_DIR/trained_eval.json"
echo "  Training log: $OUTPUT_DIR/training_log.csv"
echo "  Report PNG  : $REWARDS_DIR/comparison_report.png"
echo "========================================"
