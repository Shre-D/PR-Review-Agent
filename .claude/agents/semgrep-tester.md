---
name: semgrep-tester
model: claude-haiku-4-5
description: Test the semgrep MCP tool on a diff file. Used during manual env testing, NOT during training rollouts.
tools: Bash
---

Given $DIFF_PATH, run one semgrep scan. One shot only.

1. Run: semgrep --config=auto "$DIFF_PATH" --json 2>/dev/null || echo '{"results":[]}'
2. Count findings by severity from the JSON
3. Compute score: start 1.0, subtract 0.15 per ERROR, 0.05 per WARNING, floor 0.0

Return JSON:
{"tool": "semgrep", "score": 0.85, "critical": 0, "warning": 0, "ms": 0}
