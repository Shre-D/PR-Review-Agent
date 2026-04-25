# Results and Baselines

The trained-model result is intentionally marked pending until a LoRA checkpoint
is produced by HF Jobs or HPC and evaluated on the same 78-task benchmark.

The current submission baseline artifact is:

```text
rewards/baseline_eval.json
```

It was generated with:

```bash
PR_REVIEW_TOOL_BACKEND=heuristic \
python benchmarks/evaluate_baselines.py \
  --task-bank all \
  --task-loader-mode short \
  --output rewards/baseline_eval.json
```

## Current Baselines

| Policy | Episodes | Accuracy | Mean Return |
|---|---:|---:|---:|
| Random | 78 | 0.449 | 0.821 |
| Heuristic | 78 | 0.590 | 2.505 |
| Trained SLM | pending | pending | pending |

## What the Baselines Mean

The random baseline chooses a random analysis tool, then submits a verdict using
the baseline verdict function. It is useful as a lower bound: any trained router
must do better than accidental tool choice.

The heuristic baseline is stronger. It uses file types and language rules:

- always call `check_security`
- call `check_config` for Dockerfile, YAML, `.gitignore`, and GitHub Actions
- call `check_quality` for source languages where quality issues are common
- call `check_build_and_types` for Java, Go, Rust, build manifests, and workflows
- call `check_tests` for languages where behavior changes should have coverage

Then it submits a verdict based on tool scores.

This is the baseline the trained SLM must beat.

## Why Accuracy Is Not Enough

Accuracy only measures final verdict correctness. Mean return also measures the
route quality:

- Did the policy gather evidence?
- Was the evidence relevant?
- Did the policy avoid redundant tools?
- Did the final verdict align with the findings?

A model that guesses correct verdicts without tool evidence can have acceptable
accuracy but poor return. The intended model should improve both.

## Result Artifact Format

Baseline and trained evaluation artifacts include:

- policy name
- episode count
- accuracy
- mean episode return
- breakdown by language
- breakdown by file type
- example episodes
- duplicate tool rate
- invalid action rate
- early submit rate
- over-budget rate
- evidence-backed verdict rate
- mean steps by route tier
- mean tool calls
- context-dependent task metrics when tagged tasks are present

Trained evaluation is generated with:

```bash
PR_REVIEW_TOOL_BACKEND=heuristic \
python benchmarks/evaluate_trained_model.py \
  --checkpoint ./grpo_checkpoint \
  --preset a100 \
  --task-bank all \
  --task-loader-mode short \
  --output rewards/trained_eval.json
```

## Expected Final Comparison

The final report should compare:

1. random router
2. heuristic router
3. trained `Qwen/Qwen3-1.7B` LoRA router

After training, generate the plot:

```bash
python benchmarks/generate_report.py \
  --baseline rewards/baseline_eval.json \
  --trained rewards/trained_eval.json \
  --log grpo_checkpoint/training_log.csv \
  --output rewards/comparison_report.png
```

## What Counts as Success

The trained model is successful if it:

- beats random by a wide margin
- beats or meaningfully approaches the heuristic on accuracy
- improves mean return by learning shorter or better-supported routes
- produces valid JSON reliably
- handles comprehensive multi-file tasks without collapsing into one-size-fits-all
  security checks

The strongest result would be a model that beats the heuristic on mean return,
even if accuracy is close. That would show it learned route efficiency rather
than merely imitating static rules.
