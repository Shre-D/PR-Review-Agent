# Architecture Decisions Log

Format: [TYPE] Description — timestamp
Types: INIT, ARCH, BUILD, RESEARCH, TEST, DEPLOY, DEBUG

## Initial decisions
[INIT] Repo scaffolded with OpenEnv directory layout — $(date '+%Y-%m-%d')
[ARCH] Using MCPEnvironment base class with FastMCP for tool registration — $(date '+%Y-%m-%d')
[ARCH] Static tools: semgrep (security), pylint (lint), radon (complexity) — $(date '+%Y-%m-%d')
[ARCH] Base model: Qwen/Qwen3-1.7B (matches official OpenEnv tutorial) — $(date '+%Y-%m-%d')
[ARCH] Reward: dense per-informative-tool-call + redundancy penalty + step penalty — $(date '+%Y-%m-%d')
[ARCH] Training: TRL GRPOTrainer via HTTPEnvClient, not custom loop — $(date '+%Y-%m-%d')
[ARCH] Deploy: openenv push to HF Spaces (NOT openenv deploy, which does not exist) — $(date '+%Y-%m-%d')
