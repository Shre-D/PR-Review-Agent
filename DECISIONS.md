# Architecture Decisions

Format: `[TYPE] description — date`

## Final Submission

[ARCH] Scoped the submission to a single multi-language PR-review routing benchmark, not a software-team simulator — 2026-04-24
[ARCH] Supported code languages are Python, TypeScript/JavaScript, Java, Go, and Rust — 2026-04-24
[ARCH] Supported repo-global file types are Dockerfile, YAML, GitHub Actions workflow YAML, and `.gitignore` — 2026-04-24
[ARCH] Runtime action set is `check_security`, `check_quality`, `check_build_and_types`, `check_tests`, `check_config`, `submit_review`, `escalate` — 2026-04-24
[ARCH] Primary policy model is Qwen/Qwen3-1.7B; Claude/Codex/Gemini remain build-time tooling only — 2026-04-24
[ARCH] Tool execution uses heuristic-first analysis with optional hybrid use of semgrep, pylint, and radon when available — 2026-04-24
[ARCH] Added fixture-backed workspaces so representative tasks can run real analyzers and compilers against actual files — 2026-04-24
[ARCH] Added a repo-local compatibility layer so the env remains importable despite the current upstream OpenEnv dependency mismatch — 2026-04-24
[ARCH] Seeded the benchmark with curated multi-language and config-heavy PR tasks plus baseline evaluation scripts — 2026-04-24
[BUILD] envs/pr_review_env/models.py — complete — 2026-04-24 (PRReviewAction, PRReviewObservation, PRReviewState; 42 lines; import verified clean)
[BUILD] envs/pr_review_env/server/tasks.py — complete — 2026-04-24 (PRTask dataclass, 5 loader fns, 12 fallback tasks, TASKS_PATH via __file__; 350 lines; import verified clean)
[BUILD] train/run_baseline.py — complete — 2026-04-24 (thin wrapper over evaluate_baselines.evaluate_policy, ASCII summary table, smoke run verified: heuristic acc=0.385 mean=0.166, random acc=0.354 mean=-0.135; 74 lines)
[BUILD] train/grpo_train.py — complete — 2026-04-24 (GRPO loop, Qwen3-1.7B, parse_action+rollout_episode+collect_trajectories, ML deps guarded in try/except, imports clean without trl; 203 lines)
[BUILD] envs/pr_review_env/client/pr_review_env_client.py — complete — 2026-04-24 (PRReviewEnvClient(EnvClient[...]), _parse_result handles envelope+flat, __init__.py exports verified; 38 lines)
[BUILD] envs/pr_review_env/server/Dockerfile + requirements.txt — complete — 2026-04-24 (added missing pyyaml dep; replaced curl HEALTHCHECK with python urllib; pyyaml added to pyproject.toml server extras)
[BUILD] envs/pr_review_env/server/app.py + pr_review_env.py — complete — 2026-04-24 (PRReviewEnv(MCPEnvironment), create_fastapi_app wired; reset+step+state verified; full episode cycle tested)
[BUILD] envs/pr_review_env/server/tools.py — complete — 2026-04-24 (7 MCP tools, hybrid heuristic+real-tool backend; fixed gitignore narrowing heuristic + split command-execution patterns; removed dead _append_real_tool; 654 lines; all heuristics verified)
[BUILD] envs/pr_review_env/server/grader.py — complete — 2026-04-24 (DOMAIN_TO_TOOLS mapping, relevant_tools/step_reward/terminal_reward/aggregate_tool_scores; fixed invalid ci+clean domain refs; 129 lines; logic verified)
