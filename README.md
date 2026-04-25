# PR Review Router — SLM-as-Router for Code Review

> **Thesis.** Frontier LLMs review pull requests by guessing. We replace the LLM
> with a 1.7B router that learns *when* to call compilers and static analyzers,
> then synthesises their evidence into a verdict. The grader never reads the
> model's reasoning — only the tools it called and the verdict it returned.
> This is, to our knowledge, the first open RL environment for SLM-based tool
> routing in code review.

## Why This Matters

Every shipped PR-review tool today (CodeRabbit, Greptile, Cursor Bugbot, GitHub
Copilot review) follows the same pattern: feed the diff to a frontier LLM and
ask for prose. That has three structural problems:

1. **Findings are guesses.** The LLM hallucinates issues that aren't there and
   misses ones that are. There is no ground truth in its outputs.
2. **Cost scales with PR volume.** Frontier review of a medium PR is
   $0.10–$0.50. A repo doing 1k PRs/month spends real money on a probabilistic
   reviewer.
3. **Behaviour is not trainable.** You can't fine-tune the reviewer to your
   repo's policies without changing the model.

This project takes the opposite stance:

- **Compilers don't hallucinate.** `semgrep`, `ruff`, `tsc`, `javac`, `go vet`,
  `cargo check` produce evidence that is correct by construction. The router's
  job is to call the right one, not to re-derive it.
- **A 1.7B model is enough for the routing decision.** Choosing between five
  analyzers given a diff is not a frontier-scale problem. Qwen3-1.7B at 4-bit
  quantisation runs ~50 tok/s on a single T4 and ~$0.0001 per review at
  inference cost.
- **Routing is RL-trainable with deterministic rewards.** Because tool outputs
  and the expected verdict are both deterministic, GRPO has a clean signal.

The research bet: **structured action spaces let small models match large-model
performance on routing-shaped tasks.** This env is the testbed.

## What's in the Box

- An OpenEnv-compatible `MCPEnvironment` exposing five analysis tools and two
  terminal actions.
- 78 PR tasks across Python, TypeScript, JavaScript, Java, Go, Rust, plus
  YAML / Dockerfile / GitHub Actions / `.gitignore` files. 7 risk domains, 3
  difficulty tiers, and 7 repo-policy-aware tasks that require organisational
  context to grade correctly.
- A reward grader that is **epistemically blind to the agent's reasoning**:
  it only sees tool calls, tool results, and the submitted verdict. This makes
  reward gaming via persuasive prose impossible.
- A working TRL GRPO + QLoRA training pipeline targeting `Qwen/Qwen3-1.7B`,
  with hardware presets for T4, V100, A100, and H100.
- Three-platform training: Colab notebook, Hugging Face Jobs, Slurm/HPC.
- Random and heuristic baseline policies for comparison.

## Hackathon Materials

- **Hugging Face Space**: _coming after onsite training_ — `https://huggingface.co/spaces/<user>/pr-review-env`
- **Trained model**: _coming after onsite training_ — `https://huggingface.co/<user>/pr-review-qwen3-1p7b`
- **Mini-blog**: see [`blog/`](blog/) — 7 posts covering motivation, architecture,
  reward design, RL loop, tasks, tools, and results.
- **Demo video** (≤2 min): _to be linked here after recording_.
- **Slides**: _to be linked here_.
- **Theme**: Wild Card (#5) — *SLM-as-router for tool-grounded code review*.

## Repository Layout

| Path | Purpose |
|---|---|
| `envs/pr_review_env/` | OpenEnv environment, models, server, MCP tools, grader, client |
| `tasks/` | 78-task benchmark + builders + per-task context metadata |
| `train/grpo_train.py` | TRL GRPO + QLoRA training entrypoint |
| `train/train_config.py` | Hardware presets (cpu/t4/v100/a100/h100) |
| `train/adaptive_router.py` | Tier selection + per-prompt route-context injection |
| `benchmarks/evaluate_baselines.py` | Random + heuristic policy eval |
| `benchmarks/evaluate_trained_model.py` | Trained-LoRA-checkpoint eval |
| `benchmarks/generate_report.py` | 4-panel comparison plot from logs and eval JSONs |
| `colab_training.ipynb` | One-click Colab training notebook |
| `scripts/run_hf_job.sh`, `scripts/hf_train_job.py` | Hugging Face Jobs launcher |
| `hpc/train_h100.slurm`, `hpc/setup_env.sh` | Slurm/HPC training |
| `ui/gradio_app.py` | Task explorer + heuristic smoke UI |
| `blog/` | 7-post mini-blog explaining the project |
| `docs/` | Architecture, design spec, review-tool reference, company guidelines |

## How the Environment Works

Every episode starts with a `PRTask`: a diff, PR description, primary language,
changed file types, expected verdict, and labelled risk domains.

The policy emits one JSON tool call at a time:

```json
{"tool_name": "check_security", "arguments": {}}
```

Available tools:

| Tool | Backed by (heuristic / hybrid) |
|---|---|
| `check_security` | regex + AST patterns / `semgrep` |
| `check_quality` | regex + AST / `ruff`, `pylint`, `radon` |
| `check_build_and_types` | manifest patterns / `tsc`, `javac`, `go test`, `cargo check`, `pyright` |
| `check_tests` | diff/fixture inspection |
| `check_config` | regex / `pyyaml`, Dockerfile rules |
| `submit_review` | terminal — verdict ∈ `{approve, request_changes, reject}` + confidence |
| `escalate` | terminal — hand off to human reviewer |

Tool backend is selected by `PR_REVIEW_TOOL_BACKEND`:

- `heuristic` (default for training): deterministic pattern matching, no
  subprocesses, fully reproducible.
- `hybrid` (default for inference): try real analysers, fall back to heuristics.

The agent is capped at 8 inference-time steps. Low-confidence submits are
redirected back to gather more evidence.

## How the Reward Works

The grader is in [`envs/pr_review_env/server/grader.py`](envs/pr_review_env/server/grader.py).
It reads only:

- which tools were called (and in what order, with what duplicates)
- the structured tool results (`score`, `findings`, `severity`)
- the submitted verdict and the task's `expected_verdict`
- the task's `risk_domains` for relevance weighting

It does **not** read the agent's chain of thought, prose reasoning, or stated
confidence. This makes reward gaming via fluent reasoning impossible.

Reward composition per episode:

- **Step reward**: `+0.05` base for any new tool call; `+0.08` bonus if the
  tool covers one of the task's risk domains; `−2.55` for duplicates;
  efficiency decay after a tier-aware step threshold.
- **Terminal reward**: `+0.95` to `+1.20` for a correct verdict supported by
  evidence; `−1.45` for a correct verdict with no supportive tool;
  `−2.10` for a correct verdict submitted before the route's `min_tools`;
  `−0.80` for escalating instead of rejecting; `−2.65` for a wrong verdict.
- **Evidence penalties**: `−0.40` for approving when critical findings exist;
  `−0.20` for rejecting when all evidence is clean.
- **Normalisation**: raw reward in `[−2.83, +1.51]` is mapped to `[0.01, 0.99]`
  for stable GRPO advantages.

## Current Baselines (78 tasks, heuristic backend)

Generated with:

```bash
PR_REVIEW_TOOL_BACKEND=heuristic \
python benchmarks/evaluate_baselines.py \
  --task-bank all --task-loader-mode short \
  --output rewards/baseline_eval.json
```

| Policy | Episodes | Accuracy | Mean Return |
|---|---:|---:|---:|
| Random | 78 | 0.449 | 0.774 |
| Heuristic | 78 | 0.590 | 2.514 |
| **Trained SLM** | _pending onsite training_ | _pending_ | _pending_ |

Baseline artifact: [`rewards/baseline_eval.json`](rewards/baseline_eval.json).

After onsite training, the trained model row will be filled in and
[`rewards/comparison_report.png`](rewards/comparison_report.png) will be
generated automatically.

## Quickstart

```bash
pip install -e ".[server,train,dev]"

# Verify the deterministic path
PR_REVIEW_TOOL_BACKEND=heuristic pytest

# Inspect the task bank
python tasks/build_task_bank.py --print-summary

# Run the API server (OpenEnv-compatible FastAPI)
PR_REVIEW_TOOL_BACKEND=heuristic \
uvicorn envs.pr_review_env.server.app:app --reload --port 8000

# Run the smoke UI
python ui/gradio_app.py
```

## Training

A100 (Hugging Face Jobs or rented):

```bash
PR_REVIEW_TOOL_BACKEND=heuristic \
python train/grpo_train.py \
  --preset a100 --task-bank all --task-loader-mode short \
  --task-limit 78 --epochs 3 --num-generations 8 \
  --report-to wandb \
  --output-dir ./grpo_checkpoint
```

T4 (Colab):

```bash
PR_REVIEW_TOOL_BACKEND=heuristic \
python train/grpo_train.py \
  --preset t4 --task-bank all --task-limit 40 --epochs 1
```

Full HF Jobs and HPC instructions live in [TRAINING.md](TRAINING.md). The
Colab notebook is at [`colab_training.ipynb`](colab_training.ipynb).

## Evaluating a Checkpoint

```bash
PR_REVIEW_TOOL_BACKEND=heuristic \
python benchmarks/evaluate_trained_model.py \
  --checkpoint ./grpo_checkpoint \
  --preset a100 --task-bank all --task-loader-mode short \
  --output rewards/trained_eval.json

python benchmarks/generate_report.py \
  --baseline rewards/baseline_eval.json \
  --trained  rewards/trained_eval.json \
  --log      grpo_checkpoint/training_log.csv \
  --output   rewards/comparison_report.png
```

## What Makes This Submission Different

- **No production tool routes a small model through real analysers.** Every
  shipped product one-shots a frontier LLM. We are training the missing
  alternative.
- **The grader is epistemically blind.** It cannot be gamed by chain-of-thought.
  This is unusual and is the strongest correctness guarantee in the design.
- **Tool outputs are compiler-grounded.** Findings come from `semgrep`, `ruff`,
  `tsc`, `javac`, `cargo check` — not from another LLM judge. There is no
  hallucination layer.
- **Training is reproducible end-to-end** on free-tier compute (Colab T4)
  through to research compute (HPC H100), with the same code path.
- **Repo-policy-aware tasks** require the agent to use organisational context
  (architecture summary, custom rules, critical paths) to grade correctly.

## Citation / Future Work

Open research questions this env makes tractable:

- How small can the router go before routing degrades? (1.7B → 0.5B → 0.1B)
- Does an SFT warm-start from heuristic traces dominate cold GRPO?
- Does the routing policy transfer to non-PR domains (incident triage,
  test selection, data-pipeline failure routing)?
- How does the trained 1.7B router compare against frontier-LLM one-shot
  review on accuracy, latency, and cost-per-review?

If you use this environment, please link back to the Hugging Face Space.

## Held-Out Evaluation

13 task IDs in [`tasks/holdout_ids.txt`](tasks/holdout_ids.txt) are excluded
from training and reserved for measuring generalisation. The training script
reads the file automatically; the evaluator can slice to either side:

```bash
# trained-only slice (65 tasks the model saw)
python benchmarks/evaluate_trained_model.py --checkpoint ./grpo_checkpoint \
  --eval-mode train_only --output rewards/trained_eval_train.json

# held-out slice (13 unseen tasks — generalisation signal)
python benchmarks/evaluate_trained_model.py --checkpoint ./grpo_checkpoint \
  --eval-mode holdout_only --output rewards/trained_eval_holdout.json
```

A trained router that beats the heuristic on both the train slice and the
holdout slice has actually learnt routing, not memorisation.

## Limitations (What This Env Does Not Do)

- **Synthetic diffs.** Tasks are hand-authored, not scraped real PRs. This
  buys determinism for GRPO at the cost of distribution realism.
- **Heuristic backend during training.** Real `semgrep`/`ruff`/`tsc` are
  available via `PR_REVIEW_TOOL_BACKEND=hybrid`, but training runs with
  `heuristic` for reproducibility. Inference can use the real backends.
- **Single-turn observation per step.** No long-context reasoning across
  files; the model sees the diff plus prior tool results only.
- **78 tasks is small** for claims about generalisation across all of code
  review. We treat it as a benchmark, not a frontier evaluation.
- **No human reviewer comparison.** Verdicts are graded against authored
  ground truth, not against expert human reviewers. Some tasks have a single
  defensible verdict; others are intentionally ambiguous and graded for the
  *evidence path*, not the prose.

These are deliberate scoping choices. They're listed here because research
maturity means naming your limits before someone else does.

## Cross-Domain Transfer (Future Work)

The same routing framework — small policy + structured action space + tool
results graded blind — should transfer to other domains where evidence is
deterministic and routing is the hard part:

- **Incident triage**: route an alert through `check_logs`, `check_metrics`,
  `check_traces`, `check_recent_deploys` → submit severity + on-call handoff.
- **Test selection**: route a diff through `test_impact_analysis`,
  `flaky_history`, `dependency_graph` → submit test-subset to run.
- **Data-pipeline failure routing**: route a job failure through
  `check_schema_drift`, `check_upstream_volume`, `check_quotas` → submit
  fix path or escalate.

These share the same structural property as PR review: the LLM doesn't need
to *know* the answer — it needs to know *which deterministic check answers
the question*. We expect the trained 1.7B router to transfer with light
domain-specific finetuning.

## Artifact Policy

Do not commit checkpoints, W&B runs, tokens, virtual environments, or large
generated outputs. LoRA adapters belong in `grpo_checkpoint/`, HPC scratch, or
the configured Hugging Face model repository.
