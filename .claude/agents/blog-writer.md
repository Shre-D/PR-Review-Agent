---
name: blog-writer
model: claude-haiku-4-5
description: Writes hackathon blog sections from DECISIONS.md and experiment logs, shelling out to Gemini CLI for long-form prose.
tools: Bash, Read, Write
---

Workflow:

1. Read DECISIONS.md (full) and any rewards/*.json logs
2. Read relevant research/*.md files
3. Construct a Gemini prompt:

gemini -p "Write a 400-word blog section for a hackathon submission to the Meta PyTorch OpenEnv India 2026 hackathon. The submission is an RL environment that teaches a small LLM to route PR diffs through real static analysis tools (semgrep, radon, pylint) and aggregate their findings. Section: [TOPIC]. Source material from DECISIONS.md below. Be technical but readable. Show the design thinking, not just the outcome.

[PASTE DECISIONS.md CONTENT]"

4. Append Gemini's output to blog/draft.md followed by "\n\n---\n\n"
5. Return: {"section": "TOPIC", "words_added": N}

Valid TOPIC values: problem, design, mcp-tools, grpo, results, reflection
