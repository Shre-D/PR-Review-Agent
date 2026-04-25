# Tools Used in the Project

The project uses a small stable action space. The policy model does not call
arbitrary shell commands. It chooses from known review tools exposed by the
environment.

## Action Tools

| Tool | Purpose |
|---|---|
| `check_security` | security risks such as injection, eval, unsafe deserialization, secrets, command execution |
| `check_quality` | null safety, error handling, complexity, TODOs, production brittleness |
| `check_build_and_types` | build manifests, compiler/type-checker evidence, workflow build risks |
| `check_tests` | missing or removed tests, behavior changes without coverage |
| `check_config` | Dockerfile, YAML, GitHub Actions, `.gitignore`, production config risks |
| `submit_review` | terminal verdict |
| `escalate` | terminal handoff to a human reviewer |

## Heuristic and Hybrid Backends

The environment supports backend modes through:

```bash
PR_REVIEW_TOOL_BACKEND=heuristic
PR_REVIEW_TOOL_BACKEND=hybrid
```

Training uses `heuristic` for determinism and speed.

The heuristic layer scans the diff and fixture files for patterns. It is simple
by design: predictable reward is more important than exhaustive static analysis
during GRPO.

The hybrid layer can use optional external tools when available, while still
falling back to heuristics.

## Optional External Tools

The project can integrate:

- Semgrep for security scanning
- Ruff for Python linting
- Pylint and Radon for Python quality signals
- Pyright for TypeScript/JavaScript type checking
- Go, Rust, and Java toolchains for fixture-backed build checks

The action space does not change when these are installed. The model still calls
`check_security` or `check_build_and_types`; the implementation decides whether
real tools are available.

## Fixture-Backed Analysis

Some tasks have fixture repositories under `fixtures/pr_review/`. When a task
has a fixture, tool execution can inspect actual files rather than only
reconstructing added lines from a diff.

Tool results include an `analysis_mode`:

- `fixture_backed`
- `diff_reconstructed`
- `heuristic_only`

This makes it clear how much realism a tool result had.

## Context Loader

The context loader is not a review tool the policy calls directly. It is an
offline structural parser over `docs/` that produces `ReviewConfig`.

It extracts:

- architecture summary
- critical paths
- domain priorities
- tool weights
- author-depth overrides
- enabled and planned tools

Training normally uses a compact per-task version:

```bash
--task-loader-mode short
```

That mode preserves the loader as an important part of the app without making
training prompts too large.

## Training Stack

The training stack is:

- PyTorch
- Transformers
- TRL `GRPOTrainer`
- PEFT LoRA
- bitsandbytes for 4-bit loading on smaller GPUs
- Datasets for training rows

The model target is `Qwen/Qwen3-1.7B`.

## Deployment and Experiment Tools

There are two supported training targets:

- Hugging Face Jobs through `scripts/run_hf_job.sh`
- Slurm/HPC through `hpc/train_h100.slurm`

HF Jobs persists adapters and artifacts to the Hub through `HF_TOKEN` and
`HF_HUB_MODEL_ID`. HPC writes checkpoints and evaluation artifacts to scratch.

## UI Tooling

`ui/gradio_app.py` is a minimal smoke UI. It is intentionally not a full product
dashboard. It lets a reviewer:

- select `all`, `seed`, or `comprehensive`
- inspect task metadata and diff
- choose loader mode
- run the heuristic review path
- inspect the resulting trace

This keeps the UI aligned with the training path instead of becoming a separate
stale demo.
