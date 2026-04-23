# PR Review RL — Agent Constitution

## What we're building
An OpenEnv-compatible RL environment where a small LLM learns to route PR diffs
through real static analysis tools (semgrep, radon, pylint) exposed as MCP tools,
then aggregate tool outputs into a merge/reject/escalate decision.

- Framework: OpenEnv (meta-pytorch), MCPEnvironment base class
- MCP framework: FastMCP (tools defined via @mcp.tool decorators)
- Transport: FastAPI HTTP server in Docker container
- Training: TRL GRPOTrainer via HuggingFace TRL-OpenEnv integration
- Base model: Qwen/Qwen3-1.7B (matches official tutorial)
- Deploy: `openenv push` to Hugging Face Spaces
- Hackathon: Meta PyTorch OpenEnv India 2026, finale April 25-26

## Directory contract (DO NOT deviate)
envs/pr_review_env/
├── models.py                 # Pydantic Action/Observation/State
├── client/
│   └── pr_review_env_client.py   # HTTPEnvClient subclass
└── server/
    ├── app.py                # create_fastapi_app entrypoint
    ├── pr_review_env.py      # MCPEnvironment subclass
    ├── tools.py              # FastMCP @mcp.tool functions (semgrep/radon/pylint)
    ├── grader.py             # Reward aggregation
    ├── tasks.py              # PR task loader
    ├── requirements.txt
    └── Dockerfile

## Hard rules — never break
1. All Python code is written via subagent delegation (codex-writer or claude-writer).
2. After every architectural decision, append ONE line to DECISIONS.md immediately.
3. Spoken responses: under 80 words unless the human asks for more.
4. Read-only recon tasks use Haiku. Code writing uses Sonnet. Research uses Gemini CLI.
5. NEVER call heavy LLMs (senior-engineer style) inside the training rollout — it balloons cost.
6. Run `openenv validate <env_path>` before every `openenv push`.
7. Before writing env code, always read .claude/skills/openenv-api/SKILL.md first.

## Explore → Plan → Execute → Verify (mandatory for every /build)
1. EXPLORE: spawn pr-explorer (Haiku) → returns JSON facts about existing files
2. PLAN: append SPEC entry to SPEC.md
3. EXECUTE: spawn codex-writer OR shell out to `codex exec` / `gemini -p`
4. VERIFY: re-spawn pr-explorer → confirm file exists, imports clean
5. LOG: append to DECISIONS.md

## Critical corrections from outdated blueprints
- OpenEnv is NOT a gym.Env wrapper. It is FastAPI + Docker + Pydantic models.
- `openenv validate` exists; `openenv deploy` does NOT — use `openenv push`.
- MCP tools are defined via `@mcp.tool` decorators on a FastMCP server instance,
  NOT via custom subprocess wrappers.
- Training uses TRL's rollout_func pattern with the env client — not a custom loop.
- Tools must NOT use reserved names: reset, step, state, close (will raise ValueError).

## Judging criteria (keep in mind always)
1. Environment Innovation (40%) — novel tool-routing domain, real SAST, not toy
2. Storytelling (30%) — blog/DECISIONS.md explains the routing problem clearly
3. Reward improvement (20%) — training curve vs. random/rule-based baselines
4. Pipeline setup (10%) — openenv push succeeds, TRL training script runs

## Token budget law
- Read-only scoring / exploration: Haiku subagents ONLY
- Env code, grader logic, reward aggregation: Sonnet (main or delegated)
- Research, literature surveys: Gemini CLI via bash
- Blog writing: Gemini CLI (long-form, cheap, good prose)
- Context budget: if >70% full, run /compact before spawning more agents

## Non-negotiable technical choices
- Base model: Qwen/Qwen3-1.7B (works on free Colab T4)
- Task bank: minimum 50 synthetic PR diffs before training starts
- Action set: 5 MCP tools - call_security, call_linter, call_complexity, submit_review, escalate
- Reward: dense per-informative-tool-call, redundancy penalty, step penalty after step 3
