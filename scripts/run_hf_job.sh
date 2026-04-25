#!/usr/bin/env bash
# Launch GRPO training on Hugging Face Jobs from the repository root.
#
# Prereqs:
#   hf auth login
#   export REPO_URL="https://github.com/<user>/<repo>.git"
#   export HF_HUB_MODEL_ID="username/pr-review-qwen3-1p7b"
#
# HF Jobs bills by minute while the job is starting/running. Set --timeout
# explicitly so training is not killed by the default timeout.

set -euo pipefail

HF_HUB_MODEL_ID="${HF_HUB_MODEL_ID:?Set HF_HUB_MODEL_ID, for example username/pr-review-qwen3-1p7b}"
REPO_URL="${REPO_URL:?Set REPO_URL to the git URL for this repository}"
REPO_REF="${REPO_REF:-main}"
FLAVOR="${FLAVOR:-a100-large}"
TIMEOUT="${TIMEOUT:-3h}"
PRESET="${PRESET:-a100}"
TASK_LIMIT="${TASK_LIMIT:-78}"
EPOCHS="${EPOCHS:-3}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
TASK_LOADER_MODE="${TASK_LOADER_MODE:-short}"

hf jobs uv run \
  --python 3.11 \
  --flavor "$FLAVOR" \
  --timeout "$TIMEOUT" \
  --secrets HF_TOKEN \
  --env PR_REVIEW_TOOL_BACKEND=heuristic \
  scripts/hf_train_job.py \
    --repo-url "$REPO_URL" \
    --repo-ref "$REPO_REF" \
    --hub-model-id "$HF_HUB_MODEL_ID" \
    --preset "$PRESET" \
    --task-loader-mode "$TASK_LOADER_MODE" \
    --task-limit "$TASK_LIMIT" \
    --epochs "$EPOCHS" \
    --num-generations "$NUM_GENERATIONS"
