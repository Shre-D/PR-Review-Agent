# Training Guide

This repository trains a `Qwen/Qwen3-1.7B` PR-review router with TRL GRPO and
QLoRA. Training is SLM-only: no larger model is called during rollout, reward
calculation, routing, or evaluation. The structural loader is used only to build
short per-task review context.

## Current Benchmark

- Default task bank: `--task-bank all`
- Total tasks: `78`
- Composition: `65` seed tasks plus `13` comprehensive multi-file tasks
- Default loader mode: `--task-loader-mode short`
- Deterministic tool backend for training: `PR_REVIEW_TOOL_BACKEND=heuristic`

Current baseline artifact: `rewards/baseline_eval.json`

| Policy | Episodes | Accuracy | Mean Return |
|---|---:|---:|---:|
| Random | 78 | 0.449 | 0.821 |
| Heuristic | 78 | 0.590 | 2.505 |
| Trained SLM | pending | pending | pending |

The reward scale is now strictly positive at the exposed training/evaluation
boundary. Baseline artifacts also include duplicate tool rate, invalid action
rate, early submit rate, evidence-backed verdict rate, mean steps by route tier,
and context-dependent task results.

## Local Smoke Checks

```bash
pip install -e ".[server,train,dev]"
python tasks/build_task_bank.py --print-summary
PR_REVIEW_TOOL_BACKEND=heuristic pytest
```

CPU smoke test for the training data/reward path:

```bash
PR_REVIEW_TOOL_BACKEND=heuristic \
python -c "from train.train_config import TrainingConfig; from train.grpo_train import build_training_state_rows; cfg=TrainingConfig.for_cpu(); cfg.tasks_file='all'; cfg.training_task_limit=2; print(len(build_training_state_rows(cfg)))"
```

## Hugging Face Jobs

HF Jobs is the preferred paid cloud path. It requires a logged-in HF account
with prepaid credits and a token that can create/manage jobs and write to the
target model repository.

Prereqs:

```bash
pip install -U "huggingface_hub[cli]"
hf auth login
export REPO_URL="https://github.com/<user>/<repo>.git"
export REPO_REF="main"
export HF_HUB_MODEL_ID="<user>/pr-review-qwen3-1p7b"
```

Launch an A100 run:

```bash
bash scripts/run_hf_job.sh
```

Useful overrides:

```bash
FLAVOR=a100-large TIMEOUT=3h PRESET=a100 EPOCHS=3 TASK_LIMIT=78 NUM_GENERATIONS=8 \
bash scripts/run_hf_job.sh
```

The HF job script:

- clones `REPO_URL`
- installs `.[server,train,dev]`
- regenerates `tasks/all_tasks.jsonl`
- trains with `--task-bank all --task-loader-mode short`
- pushes the LoRA adapter to `HF_HUB_MODEL_ID`
- uploads baseline/trained eval JSON and `training_log.csv` under `artifacts/`

Current HF Jobs pricing guidance from Hugging Face docs: Jobs bill by minute
while starting/running, require prepaid credits, and support flavors including
`t4-small`, `l4`, `a10g-large`, `a100-large`, and `h200`.

## HPC

Use this path when an H100 or equivalent Slurm partition is available.

Prepare the environment once on the login node:

```bash
bash hpc/setup_env.sh
```

Submit training from the repository root:

```bash
sbatch hpc/train_h100.slurm
```

The Slurm job uses:

- Python 3.11
- cached HF model weights in `/scratch/$USER/hf_cache`
- `--preset h100`
- `--task-bank all`
- `--task-loader-mode short`
- heuristic tool backend
- full baseline and trained evaluation after training

Override cluster-specific values through environment variables:

```bash
ENV_DIR=/scratch/$USER/envs/pr-review \
HF_CACHE=/scratch/$USER/hf_cache \
OUTPUT_DIR=/scratch/$USER/pr-review/grpo_checkpoint \
REWARDS_DIR=/scratch/$USER/pr-review/rewards \
sbatch hpc/train_h100.slurm
```

## Direct Training Command

```bash
PR_REVIEW_TOOL_BACKEND=heuristic \
python train/grpo_train.py \
  --preset a100 \
  --task-bank all \
  --task-loader-mode short \
  --task-limit 78 \
  --epochs 3 \
  --num-generations 8 \
  --output-dir ./grpo_checkpoint
```

For faster iteration:

```bash
PR_REVIEW_TOOL_BACKEND=heuristic \
python train/grpo_train.py \
  --preset t4 \
  --task-bank all \
  --task-loader-mode empty \
  --task-limit 20 \
  --epochs 1 \
  --num-generations 2 \
  --output-dir ./grpo_checkpoint_smoke
```

## Evaluation

```bash
PR_REVIEW_TOOL_BACKEND=heuristic \
python benchmarks/evaluate_baselines.py \
  --task-bank all \
  --task-loader-mode short \
  --output rewards/baseline_eval.json

PR_REVIEW_TOOL_BACKEND=heuristic \
python benchmarks/evaluate_trained_model.py \
  --checkpoint ./grpo_checkpoint \
  --preset a100 \
  --task-bank all \
  --task-loader-mode short \
  --output rewards/trained_eval.json
```

Generate the comparison plot:

```bash
python benchmarks/generate_report.py \
  --baseline rewards/baseline_eval.json \
  --trained rewards/trained_eval.json \
  --log grpo_checkpoint/training_log.csv \
  --output rewards/comparison_report.png
```

## Training Readiness Checklist

- `tasks/all_tasks.jsonl` has 78 tasks.
- Context-dependent task metadata is present for repo-policy-sensitive tasks.
- `PR_REVIEW_TOOL_BACKEND=heuristic pytest` passes.
- HF path has `REPO_URL`, `HF_HUB_MODEL_ID`, and `HF_TOKEN` available.
- HPC path has Python 3.11, CUDA-visible PyTorch, and Qwen weights cached.
- Trained checkpoints are not committed; keep them in `grpo_checkpoint/`,
  scratch storage, or the configured Hugging Face model repo.

## Common Failures

| Symptom | Likely Cause | Fix |
|---|---|---|
| `GRPO requires at least 2 generations` | `--num-generations` too low | Use `2` or higher |
| CUDA OOM | Batch/generation count too high | Use `t4`, reduce `--num-generations`, or lower `--task-limit` |
| HF job exits before training completes | default timeout | pass `TIMEOUT=3h` or larger |
| HF checkpoint missing | no Hub push token/repo | set `HF_HUB_MODEL_ID` and pass `--secrets HF_TOKEN` |
| HPC cannot import package | wrong Python env | rerun `hpc/setup_env.sh` with Python 3.11 |
