# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Setup
```bash
pip install -e ".[server,train,dev]"
```

### Tests
```bash
# All tests (heuristic backend is required for determinism)
PR_REVIEW_TOOL_BACKEND=heuristic pytest

# Single test file
PR_REVIEW_TOOL_BACKEND=heuristic pytest tests/test_grader.py

# CPU smoke test for training data/reward path
PR_REVIEW_TOOL_BACKEND=heuristic \
python -c "from train.train_config import TrainingConfig; from train.grpo_train import build_training_state_rows; cfg=TrainingConfig.for_cpu(); cfg.tasks_file='all'; cfg.training_task_limit=2; print(len(build_training_state_rows(cfg)))"
```

### Task bank
```bash
python tasks/build_task_bank.py --print-summary
```

### API server (local)
```bash
PR_REVIEW_TOOL_BACKEND=heuristic \
uvicorn envs.pr_review_env.server.app:app --reload --port 8000
```

### Baseline evaluation
```bash
PR_REVIEW_TOOL_BACKEND=heuristic \
python benchmarks/evaluate_baselines.py \
  --task-bank all \
  --task-loader-mode short \
  --output rewards/baseline_eval.json
```

### Training (A100)
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

See [TRAINING.md](TRAINING.md) for HF Jobs and HPC/Slurm instructions.

## Architecture

The system frames PR code review as a **tool-routing problem**: the policy must call the right analysis tools before submitting a verdict. The target model is `Qwen/Qwen3-1.7B` trained via TRL GRPO + QLoRA with no larger model involved at any stage.

### Environment layer (`envs/pr_review_env/`)

- **`server/pr_review_env.py`** — `PRReviewEnv(MCPEnvironment)`: the OpenEnv-compatible environment. `reset()` loads a task; `step(PRReviewAction)` dispatches tool calls via FastMCP, accumulates state, and computes rewards. Hard cap of 8 inference-time steps prevents runaway loops.
- **`server/tools.py`** — Five MCP tools registered via `@mcp.tool`: `check_security`, `check_quality`, `check_build_and_types`, `check_tests`, `check_config`. Two terminal tools: `submit_review` (verdict + confidence) and `escalate`. Each tool has a **heuristic mode** (pattern matching on diff text) and a **real-tool mode** (semgrep, ruff, pylint, radon, tsc, javac, go test, cargo check). Mode is controlled by `PR_REVIEW_TOOL_BACKEND` env var (`heuristic` | `hybrid` | default hybrid).
- **`server/grader.py`** — Epistemically independent reward grader. Never reads SLM reasoning or confidence. Computes: `step_reward` (per tool call), `terminal_reward` (on verdict), `evidence_penalties` (contradiction checks), and `normalize_reward` (maps raw `[−2.83, +1.51]` to `[0.01, 0.99]`).
- **`server/tasks.py`** — `PRTask` dataclass. Loads from `tasks/tasks.jsonl` (seed), `tasks/comprehensive_tasks.jsonl` (13 multi-file tasks), or `tasks/all_tasks.jsonl` (combined 78 tasks).
- **`server/context_loader.py`** — Structural loader that builds short per-task `ReviewConfig` from `docs/`. Loader modes: `short` (default for training), `empty`, `full`, `off`.
- **`models.py`** — Pydantic models: `PRTask`, `PRReviewState`, `PRReviewObservation`, `PRReviewAction`, `PRReviewVerdict`, `ReviewConfig`.

### Reward design

The reward function has one goal: make the policy learn that gathering the right evidence before deciding is worth more than guessing. Key values:
- Correct verdict + relevant tool called: `+1.0` to `+1.25`
- Correct verdict, no relevant tool: `+0.35` (right answer, no demonstrated process)
- Wrong verdict: `−0.55`
- Duplicate tool call: `−0.20` (per call)
- Efficiency decay: `−0.03 × (step − decay_start)` where `decay_start` is tier-aware (2/4/6)
- Evidence penalties: `−0.40` for critical findings + approve, `−0.20` for all-clean + reject

`risk_domains` on each task drive `relevant_tools()`: calling a tool that covers the task's risk domain earns an extra `+0.08` relevance bonus.

### Training layer (`train/`)

- **`grpo_train.py`** — GRPO training entrypoint. Builds prompt from `PRReviewObservation`, generates tool calls as JSON, replays them in a local `PRReviewEnv` instance to get reward. GPU presets: `cpu`, `t4`, `a100`, `h100`.
- **`train_config.py`** — `TrainingConfig` with preset factory methods.
- **`adaptive_router.py`** — `prompt_route_context()` injects task-specific routing hints into the prompt.
- **`export_teacher_traces.py`** — Exports heuristic-policy rollouts as JSONL teacher traces.

### Evaluation (`benchmarks/`)

- `evaluate_baselines.py` — Random and heuristic policy baselines.
- `evaluate_trained_model.py` — LoRA checkpoint evaluation.
- `loader_benchmark.py` — Compares loader configs.
- `generate_report.py` — Comparison plot from baseline + trained JSON + training CSV.

## Key design constraints

- `PR_REVIEW_TOOL_BACKEND=heuristic` is required for all tests and training runs (makes tool scores deterministic).
- Do not commit checkpoints, W&B runs, tokens, or `.venv/`. LoRA adapters go to `grpo_checkpoint/`, HPC scratch, or the HF model repo.
- The grader is intentionally blind to the agent's reasoning — this is a correctness requirement that prevents reward gaming.
- `_MAX_INFERENCE_STEPS = 8` in `pr_review_env.py` is a hard inference cap; low-confidence submits below `escalation_confidence_threshold` are redirected with a `−0.10` penalty.
