---
description: Deploy environment to Hugging Face Spaces via openenv push.
allowed-tools: Bash
---

Deploy pipeline:

1. Run: openenv validate envs/pr_review_env/
2. If validate fails, STOP and report errors
3. Run: cd envs/pr_review_env && openenv push --repo-id YOUR_USERNAME/pr-review-env
4. Wait for build, report the HF Space URL
5. Append to DECISIONS.md: "[DEPLOY] Pushed to HF Spaces — $(date)"

Replace YOUR_USERNAME with the actual HF username before running.
