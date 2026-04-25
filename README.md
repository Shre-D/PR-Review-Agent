# PR Review Router

OpenEnv-compatible benchmark and training path for a small PR-review router.
The central claim is that code review can be framed as a tool-routing problem:
the policy should gather the right evidence before returning a verdict.

## Submission Scope

- Policy model target: `Qwen/Qwen3-1.7B`
- Training method: TRL GRPO with QLoRA
- Task bank: `78` PR-review tasks via `--task-bank all`
- Task mix: `65` seed tasks plus `13` comprehensive multi-file tasks
- Loader: deterministic structural loader with short per-task context
- Training targets: Hugging Face Jobs and Slurm/HPC
- No larger-model calls during training, reward calculation, routing, or eval

## Repository Layout

| Path | Purpose |
|---|---|
| `envs/pr_review_env/` | environment models, server, tools, grader, and client |
| `tasks/all_tasks.jsonl` | default combined 78-task bank |
| `tasks/tasks.jsonl` | seed task bank |
| `tasks/comprehensive_tasks.jsonl` | 13 larger multi-file tasks |
| `train/grpo_train.py` | GRPO/QLoRA training entrypoint |
| `benchmarks/evaluate_baselines.py` | random and heuristic baseline eval |
| `benchmarks/evaluate_trained_model.py` | LoRA checkpoint evaluation |
| `benchmarks/loader_benchmark.py` | post-training loader/config comparison |
| `hpc/` | Slurm setup and H100 training scripts |
| `scripts/hf_train_job.py` | remote HF Jobs training entrypoint |
| `scripts/run_hf_job.sh` | local HF Jobs launcher |
| `ui/gradio_app.py` | minimal task explorer and heuristic smoke UI |

## Current Baselines

Generated with:

```bash
PR_REVIEW_TOOL_BACKEND=heuristic \
python benchmarks/evaluate_baselines.py \
  --task-bank all \
  --task-loader-mode short \
  --output rewards/baseline_eval.json
```

| Policy | Episodes | Accuracy | Mean Return |
|---|---:|---:|---:|
| Random | 78 | 0.449 | 0.821 |
| Heuristic | 78 | 0.590 | 2.505 |
| Trained SLM | pending | pending | pending |

The current evaluator also reports practical workflow metrics such as duplicate
tool rate, early submit rate, evidence-backed verdict rate, mean steps by route
tier, and mean tool calls.

## Local Checks

```bash
pip install -e ".[server,train,dev]"
python tasks/build_task_bank.py --print-summary
PR_REVIEW_TOOL_BACKEND=heuristic pytest
```

Run the API server:

```bash
PR_REVIEW_TOOL_BACKEND=heuristic \
uvicorn envs.pr_review_env.server.app:app --reload --port 8000
```

Run the minimal UI:

```bash
python ui/gradio_app.py
```

## Training

See [TRAINING.md](TRAINING.md) for full HF Jobs and HPC instructions.

Direct A100 command:

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

Evaluate a trained checkpoint:

```bash
PR_REVIEW_TOOL_BACKEND=heuristic \
python benchmarks/evaluate_trained_model.py \
  --checkpoint ./grpo_checkpoint \
  --preset a100 \
  --task-bank all \
  --task-loader-mode short \
  --output rewards/trained_eval.json
```

## Task Loader Interface

Every training and evaluation entrypoint accepts:

- `--task-bank all`: combined default bank
- `--task-bank seed`: seed bank only
- `--task-bank comprehensive`: larger multi-file tasks only
- `--tasks-file path/to/tasks.jsonl`: explicit task file

Loader modes:

- `short`: compact structural context, default for training
- `empty`: exercise loader plumbing with nearly no prompt cost
- `full`: full structural config for app/demo analysis
- `off`: no loader config

Short mode now preserves applicable custom rules from `docs/review-tool.md`
when they match the task domain or touched paths. Context-dependent tasks are
tagged with `context_requirements` and `expected_evidence` so evaluation can
separate generic tool routing from repo-policy-aware review.

## Artifact Policy

Do not commit checkpoints, W&B runs, tokens, virtual environments, or large
generated outputs. Persist trained LoRA adapters to the configured Hugging Face
model repo or HPC scratch storage.
