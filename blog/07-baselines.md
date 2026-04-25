---
title: "Baselines and Evaluation: Measuring What Matters"
date: 2026-04-24
tags: [evaluation, baselines, metrics, benchmarking]
---

# Baselines and Evaluation: Measuring What Matters

Before training, you need to know what you're improving over. We have two baselines: random tool routing and heuristic language-aware routing. Their numbers set the floor that GRPO training must beat.

## The Two Baselines

### Random Policy

The random baseline calls one randomly-selected analysis tool, then submits a verdict based on the tool scores. No language awareness, no task context.

```python
def random_policy(observation, rng) -> PRReviewAction:
    tool_name = rng.choice([
        "check_security", "check_quality", "check_build_and_types",
        "check_tests", "check_config",
    ])
    return PRReviewAction(tool_name=tool_name, arguments={"diff_str": observation.diff_str})
```

### Heuristic Policy

The heuristic baseline uses language and file type to decide which tools to call:

```python
def heuristic_policy(observation) -> list[PRReviewAction]:
    file_types = set(observation.changed_file_types)
    actions = [check_security]  # always run security
    if file_types & {"dockerfile", "yaml", "gitignore", "github_actions"}:
        actions.append(check_config)
    if observation.primary_language in {"python", "typescript", "javascript", "java", "rust"}:
        actions.append(check_quality)
    if observation.primary_language in {"java", "go", "rust"} or "build_manifest" in file_types:
        actions.append(check_build_and_types)
    if observation.primary_language in {"typescript", "javascript", "java", "python"}:
        actions.append(check_tests)
    return actions
```

## Baseline Numbers (65 Tasks, Seed=7, Heuristic Backend)

```
Policy      Tasks  Accuracy  Mean Return
----------  -----  --------  -----------
heuristic      65     0.385        0.166
random         65     0.354       -0.135
```

These numbers tell an interesting story.

## Reading the Results

**Accuracy (38.5% heuristic, 35.4% random)** — Both baselines are above chance (33% for 3 classes), but neither is good. The heuristic policy only gets the right verdict on 38.5% of tasks.

Why so low? The heuristic always runs `check_security` on every diff, regardless of context. For a clean Go refactor, that's a wasted call that doesn't help reach the correct verdict (approve) and costs efficiency budget. The verdict decision logic in `decide_final_verdict` uses score thresholds that aren't well-calibrated to all task types.

**Mean Return (+0.166 heuristic, -0.135 random)** — This is the metric that actually matters for training. The heuristic achieves positive mean return; the random policy achieves negative. This means:

1. The heuristic is *learning something useful* — its tool routing earns step bonuses and occasionally gets the right terminal reward
2. The random policy is net-negative — its random tool calls don't earn enough step reward to offset frequent wrong terminal verdicts

The gap (+0.166 vs -0.135 = **+0.301**) is the range the GRPO-trained model needs to exceed.

## What Good Looks Like

A well-trained agent should achieve:
- **Accuracy**: > 70% (nearly 2× the heuristic baseline)
- **Mean return**: > 0.8 (the average of a correct verdict with 2 relevant tools and high scores)

How? By learning:
1. **Not to call check_security on clean refactors** — saves the efficiency budget
2. **To call check_config on Dockerfile/YAML changes** — earns relevant-tool bonus
3. **To submit earlier on easy tasks** — avoids step penalties
4. **To gather 2+ relevant tools before submitting** — earns the evidence bonus in terminal reward

## Per-Language Breakdown

The full evaluation in `benchmarks/evaluate_baselines.py` breaks results down by language and file type:

```python
results = evaluate_policy("heuristic", seed=7)
# results["by_language"]["python"] -> {"episodes": N, "accuracy": 0.x, "mean_episode_return": 0.x}
# results["by_language"]["go"] -> ...
```

This breakdown is critical for identifying where the model learns fastest. We expect Python and TypeScript to have stronger baselines (more pattern coverage in the heuristics) and Go/Rust to be harder (build tool checks add variance).

## Running the Evaluation

```bash
# Quick baseline check
python train/run_baseline.py --seed 7

# Full evaluation with per-language breakdown
python benchmarks/evaluate_baselines.py --output rewards/baseline_eval.json
```

The output file is the benchmark artifact — save it before training, then re-run after training to measure improvement.

---

*Next: [From Hackathon to Production: Deploying on HuggingFace Spaces](./08-deployment.md)*
