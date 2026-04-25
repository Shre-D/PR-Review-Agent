# Developer Workflow Notes

This file documents the internal AI-assisted build workflow for the repository.
It is not part of the PR-review benchmark itself.

## What The Submission Is

The actual submission target is the environment under `envs/pr_review_env/`.

Submission scope:

- `Python`
- `TypeScript/JavaScript`
- `Java`
- `Go`
- `Rust`
- `Dockerfile`
- `YAML`, including GitHub Actions workflows
- `.gitignore`

Runtime action set:

- `check_security`
- `check_quality`
- `check_build_and_types`
- `check_tests`
- `check_config`
- `submit_review`
- `escalate`

Runtime model guidance:

- primary policy model: `Qwen/Qwen3-1.7B`
- optional comparison baseline: small Mistral model
- no Claude, Codex, or Gemini calls inside rollout or training

## What `.claude/` Is

The `.claude/` directory is developer tooling only:

- repo exploration helpers
- code-generation helpers
- research helpers
- hackathon writing helpers

Those files are workflow baggage, not product architecture.

## Ground Rules

1. Do not describe `.claude/` agents as part of the benchmark runtime.
2. Do not use any heavyweight proprietary model in the training loop.
3. Keep the benchmark story focused on tool routing and evidence aggregation.
4. Prefer one coherent benchmark over an unfinished multi-agent simulator.

## Current Architecture

The repo is organized around a single benchmark:

- multi-language PR-review tasks
- config-aware review of repo-global files
- deterministic tool wrappers
- reward shaping for correct, efficient review behavior
- baseline scripts for random and heuristic routing

If the local OpenEnv installation is broken, use `envs/pr_review_env/compat.py`
for repo-local testing until the dependency pins are corrected.
