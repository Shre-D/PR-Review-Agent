---
title: "From Hackathon to Production: Deploying on HuggingFace Spaces"
date: 2026-04-24
tags: [deployment, huggingface, docker, openenv, production]
---

# From Hackathon to Production: Deploying on HuggingFace Spaces

The PR Review Environment is designed to deploy in one command. Here's what that looks like and what a production integration would require.

## The Deployment Stack

```
┌─────────────────────────────────────┐
│     HuggingFace Space               │
│  ┌─────────────────────────────┐    │
│  │  Docker Container           │    │
│  │  ┌───────────────────────┐  │    │
│  │  │  uvicorn               │  │    │
│  │  │  envs.pr_review_env.  │  │    │
│  │  │  server.app:app       │  │    │
│  │  └───────────┬───────────┘  │    │
│  │              │ :8000         │    │
│  └──────────────┼───────────────┘   │
│                 │                   │
│  /health  /reset  /step  /state     │
└─────────────────────────────────────┘
         ↑
   Training loop
   (local GPU)
```

## The Dockerfile

```dockerfile
ARG BASE_IMAGE=openenv-base:latest
FROM ${BASE_IMAGE}

WORKDIR /app

COPY envs/pr_review_env/server/requirements.txt /tmp/requirements.txt
RUN grep -v semgrep /tmp/requirements.txt > /tmp/req_core.txt \
    && pip install --no-cache-dir -r /tmp/req_core.txt \
    && (pip install --no-cache-dir semgrep || echo "semgrep unavailable") \
    && rm /tmp/requirements.txt /tmp/req_core.txt

COPY envs/pr_review_env/ /app/envs/pr_review_env/
COPY tasks/ /app/tasks/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "envs.pr_review_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Key decisions:
- **`openenv-base:latest`**: The OpenEnv base image provides the framework dependencies
- **Best-effort semgrep**: Semgrep's pip package requires a native binary — we install it best-effort and fall back to heuristics if unavailable
- **Python urllib healthcheck**: No curl dependency on the base image

## Deploying to HuggingFace Spaces

```bash
# Build and push
openenv push --repo-id YOUR_USERNAME/pr-review-env

# The environment is now live at:
# https://YOUR_USERNAME-pr-review-env.hf.space
```

Training loops connect by setting:
```python
ENV_URL = "https://YOUR_USERNAME-pr-review-env.hf.space"
```

## Local Development

For fast iteration without Docker:

```bash
# Start the environment server
PR_REVIEW_TOOL_BACKEND=heuristic uvicorn envs.pr_review_env.server.app:app --reload --port 8000

# In another terminal, run a test episode
python -c "
import asyncio
from envs.pr_review_env.client.pr_review_env_client import PRReviewEnvClient
from envs.pr_review_env.models import PRReviewAction

async def main():
    async with PRReviewEnvClient('http://localhost:8000') as env:
        result = await env.reset()
        obs = result.observation
        print(f'Task: {obs.task_id}, Language: {obs.primary_language}')
        
        step = await env.step(PRReviewAction(
            tool_name='check_security',
            arguments={'diff_str': obs.diff_str}
        ))
        print(f'Security score: {step.observation.last_tool_result.get(\"score\")}')
        print(f'Reward: {step.reward}')

asyncio.run(main())
"
```

## CI Integration

The simplest production integration is a GitHub Actions workflow:

```yaml
name: PR Review Agent
on: [pull_request]

jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      
      - name: Install dependencies
        run: pip install -r envs/pr_review_env/server/requirements.txt
      
      - name: Run PR review
        env:
          PR_REVIEW_TOOL_BACKEND: heuristic
        run: |
          python -c "
          from envs.pr_review_env.server.pr_review_env import PRReviewEnv
          from benchmarks.run_baselines import heuristic_policy, decide_final_verdict
          import sys, json
          
          # In real integration: fetch the actual PR diff from GitHub API
          env = PRReviewEnv()
          obs = env.reset()
          for action in heuristic_policy(obs):
              obs = env.step(action)
          final = decide_final_verdict(obs)
          obs = env.step(final)
          verdict = final.arguments['verdict']
          print(json.dumps({'verdict': verdict, 'reward': obs.reward}))
          if verdict == 'reject':
              sys.exit(1)  # block the merge
          "
```

## What Production Requires Beyond This

1. **GitHub API integration**: Fetch the real PR diff via the GitHub API rather than benchmark tasks
2. **Trained model checkpoint**: Replace the heuristic policy with the GRPO-trained `grpo_checkpoint/` model
3. **Persistent task logging**: Store every review decision for auditing and retraining
4. **Rate limiting**: The environment server can be overwhelmed by high-frequency PR activity — add a queue
5. **Multi-repo configuration**: Different repos may need different tool weights and verdict thresholds

The architecture supports all of these without fundamental changes. The environment's HTTP API, structured observations, and deterministic tool outputs make it easy to slot in a new policy model or new integration layer.

## The Path Forward

This project demonstrates that a 1.7B parameter model, trained with GRPO on a well-designed benchmark, can learn meaningful code review routing. The next steps are:

1. **Scale the task bank** to 500+ tasks across more vulnerability classes
2. **Add real PR data** from open-source repositories (with permission)
3. **Train longer** — 3 epochs is a starting point; 10+ with curriculum learning may significantly improve accuracy
4. **Add a Rust-based security scanner** for better cross-language coverage
5. **Deploy as a GitHub App** for real-world validation

The benchmark, the environment, and the training loop are all here. The code is open. The only thing missing is the GPU time.

---

*This blog series covers the full PR Review RL project, built for the Meta PyTorch OpenEnv India 2026 Hackathon.*
