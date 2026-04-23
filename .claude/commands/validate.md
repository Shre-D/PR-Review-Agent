---
description: Run openenv validate and all pytest tests.
allowed-tools: Bash
---

Validate the environment:

1. Run: openenv validate envs/pr_review_env/
2. Run: python3 -m pytest tests/ -v
3. Run: docker build -t pr-review-test envs/pr_review_env/server/ (dry-run build)
4. Report pass/fail for each step
