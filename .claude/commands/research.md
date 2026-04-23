---
description: Research a topic using Gemini CLI. Saves bullets to research/.
allowed-tools: Task, Write, Bash
---

Research: $ARGUMENTS

1. Spawn gemini-researcher with topic: "$ARGUMENTS"
2. Confirm the file saved to research/ directory
3. Append to DECISIONS.md: "[RESEARCH] $ARGUMENTS — $(date '+%Y-%m-%d %H:%M')"
