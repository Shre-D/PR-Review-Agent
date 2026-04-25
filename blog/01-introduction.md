# Introduction: SLM-as-Router for Code Review

Every shipped PR-review tool today — CodeRabbit, Greptile, Cursor Bugbot,
GitHub Copilot review — follows the same pattern: feed the diff to a frontier
LLM and ask for prose. Findings are guesses, cost scales with PR volume, and
the reviewer is not trainable to a specific repo's policies.

This project takes the opposite stance. **Compilers and static analysers don't
hallucinate.** `semgrep`, `ruff`, `tsc`, `javac`, `go vet`, and `cargo check`
produce evidence that is correct by construction. The interesting question is
not "what does the diff mean?" but "which analyser should run, given this
diff?". That is a routing decision, and routing decisions don't need a
frontier-scale model.

The PR Review Router is an OpenEnv-compatible benchmark where a 1.7B policy
model (`Qwen/Qwen3-1.7B`) learns to route a pull request through review tools
before it submits a final verdict. To our knowledge it is the first open RL
environment for SLM-based tool routing in code review.

The model does not need to be a general-purpose senior engineer. It needs to
learn evidence-routing questions like:

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

## Why a Small Model Is the Right Choice Here

Three structural reasons:

1. **Cost.** Frontier review of a medium PR is $0.10–$0.50. A 4-bit Qwen3-1.7B
   on a single T4 runs ~50 tok/s, costing ~$0.0001 per review. For repos doing
   1k+ PRs/month, the routing-with-SLM approach is two to three orders of
   magnitude cheaper.
2. **Correctness.** Tool outputs come from compilers and static analysers, not
   from another LLM judge. There is no hallucination layer between the agent
   and ground truth.
3. **Trainability.** Because tool outputs and expected verdicts are
   deterministic, the GRPO reward signal is clean and reproducible. A small
   model is enough to learn the routing decision; a large model is not needed
   to "know everything" because the analyser does the knowing.

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
| Random | 78 | 0.449 | 0.774 |
| Heuristic | 78 | 0.590 | 2.514 |
| Trained SLM | pending | pending | pending |

The final phase is training the LoRA adapter and evaluating it against the same
78-task benchmark.
