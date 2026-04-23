# PR Review RL

An RL environment where a small LLM learns to route pull request diffs through
real static analysis tools (semgrep, radon, pylint) and aggregate their findings
into a merge / request_changes / reject / escalate decision.

Built for the **Meta PyTorch OpenEnv AI Hackathon India 2026**.

## Why this matters

LLMs hallucinate vulnerabilities. Static analyzers don't — but each analyzer
only sees a slice of the problem. The interesting ML question is: given a diff,
*which tools should be run, in what order, and how do their outputs combine
into a verdict?*

We train the routing policy with GRPO. The environment provides:
- Five MCP tools (three analyzers + submit + escalate)
- Dense rewards for informative tool calls
- A task bank of 50+ labeled PR scenarios

## Quick start

```bash
# Local dev
pip install -e ".[server,dev]"
cd envs/pr_review_env/server
uvicorn app:app --reload --port 8000

# Test
python3 tests/test_episode.py
```

## Deploy

```bash
openenv validate envs/pr_review_env/
openenv push --repo-id YOUR_USERNAME/pr-review-env
```

## Train

```bash
python3 train/grpo_train.py --env-url http://localhost:8000
```

## Citation

If this helped your work, cite the OpenEnv framework:
https://github.com/meta-pytorch/OpenEnv
