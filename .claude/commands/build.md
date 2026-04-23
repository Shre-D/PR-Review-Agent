---
description: Build a component using Explore-Plan-Execute-Verify pattern.
allowed-tools: Task, Read, Write, Bash
---

Build: $ARGUMENTS

Execute in this exact order:
1. Spawn pr-explorer on the target path and any files it depends on
2. Based on pr-explorer's JSON, write a new SPEC entry in SPEC.md
3. Spawn codex-writer with the SPEC entry as input
4. Spawn pr-explorer again to verify the file exists and imports clean
5. If verify fails, spawn codex-writer again with the error appended to spec
6. Append to DECISIONS.md: "[BUILD] $ARGUMENTS — complete — $(date '+%Y-%m-%d %H:%M')"
