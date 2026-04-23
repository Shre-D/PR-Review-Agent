---
description: Final submission checklist. Validates, tests, benchmarks, blogs, commits, deploys.
allowed-tools: Task, Bash, Read
---

Run the full submission pipeline:

1. /validate
2. python3 train/run_baseline.py --policy=random --episodes=20
3. python3 train/run_baseline.py --policy=rule_based --episodes=20
4. python3 train/run_baseline.py --policy=grpo --episodes=20 (if checkpoint exists)
5. Spawn blog-writer for "results" section
6. Spawn blog-writer for "reflection" section
7. git add -A && git commit -m "submission checkpoint $(date '+%Y-%m-%d %H:%M')"
8. /push
9. Print final checklist with pass/fail per step
