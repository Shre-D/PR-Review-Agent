# Submission Readiness

## In Scope

- 78-task PR review benchmark through `--task-bank all`
- deterministic structural loader through `--task-loader-mode short`
- heuristic and random baselines in `rewards/baseline_eval.json`
- GRPO/QLoRA training for `Qwen/Qwen3-1.7B`
- HF Jobs and Slurm/HPC training paths
- minimal Gradio task explorer for smoke demos

## Removed From Submission Surface

- stale blog posts
- old internal planning/spec documents
- obsolete async rollout collection path in `train/grpo_train.py`
- Colab-specific training instructions

## Current Baseline

| Policy | Episodes | Accuracy | Mean Return |
|---|---:|---:|---:|
| Random | 78 | 0.449 | 0.774 |
| Heuristic | 78 | 0.590 | 2.514 |

## Training Gate

Before starting paid/cloud training:

```bash
python tasks/build_task_bank.py --print-summary
PR_REVIEW_TOOL_BACKEND=heuristic pytest
PR_REVIEW_TOOL_BACKEND=heuristic python benchmarks/evaluate_baselines.py --task-bank all --task-loader-mode short
```

HF Jobs requires:

- `hf auth login`
- `REPO_URL`
- `HF_HUB_MODEL_ID`
- `--secrets HF_TOKEN`
- explicit timeout, usually `TIMEOUT=3h` or higher

HPC requires:

- Python 3.11 environment from `hpc/setup_env.sh`
- CUDA-visible PyTorch
- cached `Qwen/Qwen3-1.7B` weights
- submission from repo root with `sbatch hpc/train_h100.slurm`

## Remaining Final Artifact

The final missing artifact is the trained LoRA checkpoint plus:

- `rewards/trained_eval.json`
- `grpo_checkpoint/training_log.csv`
- optional `rewards/comparison_report.png`

Do not claim trained-model performance until those files are produced by HF Jobs
or HPC and checked against the 78-task `all` benchmark.
