# Introduction: Code Review as Tool Routing

Most automated code review systems try to answer the whole review in one shot:
read the diff, write comments, maybe suggest a verdict. This project takes a
different position: review quality depends less on fluent prose and more on
choosing the right evidence.

The PR Review Router is an OpenEnv-compatible benchmark where a small policy
model learns to route a pull request through review tools before it submits a
final verdict.

The model does not need to be a general-purpose senior engineer. It needs to
learn questions like:

- Is this a security-sensitive change?
- Is the risky file a Dockerfile, workflow, or `.gitignore` rather than source
  code?
- Is this a clean refactor that should avoid unnecessary tool calls?
- Has enough evidence been gathered to approve, request changes, reject, or
  escalate?

The trained policy target is `Qwen/Qwen3-1.7B`, fine-tuned with GRPO and LoRA.
The router is intentionally small. Larger models may be used by developers
while building the project, but they are not part of training, reward
calculation, routing, or evaluation.

## What the Environment Does

Each episode starts with a benchmark PR task. The observation contains:

- the PR description
- the diff
- primary language
- changed file types
- task metadata such as author level and expected risk domains
- prior review history
- tool results gathered so far

The policy emits one JSON tool call at a time:

```json
{"tool_name": "check_security", "arguments": {}}
```

The environment executes that action, returns a reward, and updates the review
state. The episode ends when the policy calls `submit_review` or `escalate`.

Terminal example:

```json
{
  "tool_name": "submit_review",
  "arguments": {
    "verdict": "reject",
    "confidence": 0.9,
    "reasoning": "Security evidence shows string-built SQL from user input."
  }
}
```

## Current Scope

The default benchmark is `--task-bank all`, which combines:

- `65` seed tasks from `tasks/tasks.jsonl`
- `13` comprehensive multi-file tasks from `tasks/comprehensive_tasks.jsonl`

Together they form `tasks/all_tasks.jsonl` with `78` tasks.

The benchmark covers Python, TypeScript, JavaScript, Java, Go, Rust, Dockerfile,
YAML, GitHub Actions, `.gitignore`, and build manifests. It includes pure code
PRs, config-only PRs, and mixed PRs that span code plus repository-global files.

## Why Reinforcement Learning Here

A supervised model can imitate a reviewer, but this project cares about the
sequence of review decisions. Calling every tool is slow and noisy. Calling no
tools can produce unsupported verdicts. The useful behavior is in the middle:
gather enough relevant evidence, avoid redundant checks, then submit the right
verdict.

That is why the training target is a routing policy rather than a text
generator. GRPO gives the model feedback on the whole action choice, not just
whether the final sentence looks plausible.

## Submission State

The repository is now organized around:

- one default task interface: `--task-bank all`
- one loader interface: `--task-loader-mode short`
- two training targets: Hugging Face Jobs and HPC
- deterministic baseline evaluation before trained-model results are claimed

Current baselines on all 78 tasks:

| Policy | Episodes | Accuracy | Mean Return |
|---|---:|---:|---:|
| Random | 78 | 0.449 | 0.821 |
| Heuristic | 78 | 0.590 | 2.505 |
| Trained SLM | pending | pending | pending |

The final phase is training the LoRA adapter and evaluating it against the same
78-task benchmark.
