# PR Review Router

An OpenEnv-compatible benchmark where a small policy model learns to review pull
request diffs by routing them to the right evidence sources before producing a
final verdict.

This repository is scoped for a 3-day hackathon submission around one clear
claim:

**code review is a routing problem, not just a text generation problem.**

## Final Submission Scope

The benchmark covers:

- Code diffs in `Python`, `TypeScript/JavaScript`, `Java`, `Go`, and `Rust`
- Repo-global files that appear across projects:
  - `Dockerfile`
  - `YAML`, including `.github/workflows/*.yml`
  - `.gitignore`
- Three PR families:
  - pure code PRs
  - pure config / infra PRs
  - mixed PRs spanning code plus repo-global files

The environment exposes a small review action set:

- `check_security`
- `check_quality`
- `check_build_and_types`
- `check_tests`
- `check_config`
- `submit_review`
- `escalate`

## Why This Shape

A strong review policy should learn:

- when a diff needs security evidence vs quality evidence
- when config or workflow files are the real risk
- when a clean change should be merged quickly
- when uncertainty should lead to escalation instead of hallucinated confidence

The reward function therefore favors:

- correct final verdicts
- relevant tool usage
- low redundant tool use
- short, efficient review trajectories

## Repository Layout

`envs/pr_review_env/`

- `compat.py`: keeps the repo importable when the local `openenv` install is broken
- `models.py`: typed action / observation / state models
- `server/tools.py`: FastMCP tools and heuristic or hybrid analyzers
- `server/fixtures.py`: fixture workspace helpers for tool-backed analysis
- `server/tasks.py`: multi-language task bank loader
- `server/grader.py`: reward shaping and terminal scoring
- `server/pr_review_env.py`: benchmark environment
- `client/pr_review_env_client.py`: typed client wrapper

`fixtures/pr_review/`

- small fixture workspaces for representative tasks
- lets the tools run against real files instead of only reconstructed diff snippets

`tasks/tasks.jsonl`

- generated deterministic task bank across the scoped languages and config files

`tasks/build_task_bank.py`

- regenerates the benchmark task bank

`benchmarks/evaluate_baselines.py`

- evaluates heuristics and random routing across the full task bank
- writes a JSON summary to `rewards/baseline_eval.json`

`train/export_teacher_traces.py`

- exports heuristic teacher traces for offline analysis or warm-start data

`benchmarks/run_baselines.py`

- random baseline
- heuristic language-aware baseline

`.claude/`

- internal developer workflow only
- not part of training, rollout, or the submission runtime

## Local Usage

```bash
pip install -e ".[server,dev]"
python tasks/build_task_bank.py --print-summary
pytest
python benchmarks/run_baselines.py --episodes 10
python benchmarks/evaluate_baselines.py
python train/export_teacher_traces.py --limit 10
uvicorn envs.pr_review_env.server.app:app --reload --port 8000
```

To keep tests deterministic, force heuristic-only tools:

```bash
PR_REVIEW_TOOL_BACKEND=heuristic pytest
```

The default tool mode is `hybrid`: heuristics first, plus best-effort use of
`semgrep`, `pylint`, and `radon` when available.

Tool realism is now mixed but explicit:

- fixture-backed tasks emit `analysis_mode=fixture_backed`
- diff-only cases emit `analysis_mode=diff_reconstructed`
- full fallback cases emit `analysis_mode=heuristic_only`
- tool results also include `real_tools_used`

## OpenEnv Note

The checked-in environment uses OpenEnv-compatible shapes, but the local virtual
environment currently has an upstream `openenv-core` / `fastmcp` / `mcp`
version mismatch. The `compat.py` layer exists so the repo still has working
models, env logic, and tests while that dependency pin is cleaned up.

## Baselines And Training Story

The intended submission comparison is:

- random tool router
- heuristic router
- small learned policy, preferably `Qwen/Qwen3-1.7B`

Current repo-local benchmark artifacts:

- task bank size: `65`
- baseline eval artifact: `rewards/baseline_eval.json`
- trained eval artifact: `rewards/trained_eval.json` after running checkpoint evaluation
- heuristic teacher traces: `train/export_teacher_traces.py`

Claude/Codex/Gemini are build-time tooling only. They are not part of the
benchmark rollout.

## Submission Results

Current checked-in baseline artifact:

| Policy | Episodes | Accuracy | Mean Return |
|--------|----------|----------|-------------|
| Random | 65 | 0.400 | -0.029 |
| Heuristic | 65 | 0.662 | 0.651 |
| Trained SLM | pending checkpoint | pending | pending |

Training command for the final SLM artifact:

```bash
PR_REVIEW_TOOL_BACKEND=heuristic \
.venv/bin/python train/grpo_train.py \
  --preset t4 \
  --task-limit 40 \
  --epochs 2 \
  --num-generations 4 \
  --output-dir ./grpo_checkpoint
```

Evaluation command:

```bash
PR_REVIEW_TOOL_BACKEND=heuristic \
.venv/bin/python benchmarks/evaluate_trained_model.py \
  --checkpoint ./grpo_checkpoint \
  --output rewards/trained_eval.json
```

Adaptive routing is SLM-only. Route tiers change evidence requirements and tool
budget; they do not call larger LLMs.
