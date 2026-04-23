---
description: Test the PR review environment end-to-end locally via uvicorn.
allowed-tools: Task, Bash, Read
---

Test the environment locally:

1. Spawn pr-explorer to verify envs/pr_review_env/ exists and is complete
2. In a background bash call, start the server:
   cd envs/pr_review_env/server && uvicorn app:app --port 8000 &
3. Wait 3 seconds for startup
4. Run a test client script (create if not exists):
   python3 tests/test_episode.py
5. Kill the uvicorn process: pkill -f "uvicorn app:app"
6. Report the full output including reward breakdown
