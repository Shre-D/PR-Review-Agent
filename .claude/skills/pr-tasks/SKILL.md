# PR Tasks Bank Skill

## Minimum viable task bank
50 diverse PR scenarios before GRPO training. Quality > quantity.

## Task schema

```python
from dataclasses import dataclass

@dataclass
class PRTask:
    task_id: str
    diff_str: str           # realistic git diff, + and - lines
    pr_description: str     # one-sentence description
    language: str           # python | javascript | go
    expected_verdict: str   # approve | request_changes | reject
    primary_issue: str      # security | quality | clean
    difficulty: str         # easy | medium | hard
```

## 5 required archetypes (hand-authored, each ~20-50 LOC diff)

1. **SQL injection** — Python Flask endpoint concatenating user input into query. Verdict: reject. Issue: security.
2. **Clean refactor** — Python function extracted into smaller helpers, tests still pass. Verdict: approve. Issue: clean.
3. **eval() usage** — JavaScript code using eval() on user-provided string. Verdict: reject. Issue: security.
4. **High complexity** — Python function with nested if/else, CC > 15. Verdict: request_changes. Issue: quality.
5. **Dependency bump** — Python requirements.txt updating a library minor version. Verdict: approve. Issue: clean.

## Synthetic generation (45 more)

Generate with Gemini CLI:

```bash
gemini -p "Generate 45 diverse Python/JS PR diffs in JSONL format matching this schema: {task_id, diff_str, pr_description, language, expected_verdict, primary_issue, difficulty}. Include mix: 30% security issues (SQLi, XSS, pickle, eval, weak crypto, hardcoded secrets), 30% quality issues (complexity, dead code, bad names, missing tests), 40% clean changes. Vary difficulty. Keep diffs realistic, 10-40 lines." > tasks/synthetic.jsonl
```

## Format: one JSON per line in tasks/tasks.jsonl

```jsonl
{"task_id": "t001", "diff_str": "...", "pr_description": "...", "language": "python", "expected_verdict": "reject", "primary_issue": "security", "difficulty": "medium"}
{"task_id": "t002", ...}
```

## Loader

```python
import json
import random
from pathlib import Path

def load_tasks(path="tasks/tasks.jsonl"):
    with open(path) as f:
        return [PRTask(**json.loads(line)) for line in f if line.strip()]

def get_random_task():
    tasks = load_tasks()
    return random.choice(tasks)
```
