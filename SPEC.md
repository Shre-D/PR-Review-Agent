# Component Specifications

Each /build command adds a spec section here before delegating to codex-writer.
Format:
## [path/to/file.py] — status
### Purpose
One sentence.
### Requirements
- Bullet list

---

## [envs/pr_review_env/models.py] — TO BUILD
### Purpose
Pydantic dataclasses for Action, Observation, State shared by client and server.
### Requirements
- PRReviewAction(Action): tool_name: str, arguments: dict
- PRReviewObservation(Observation): diff_str, pr_description, tools_called (list), tool_results (dict), step_count (int), done (bool), reward (float)
- PRReviewState(State): task_id, expected_verdict, tools_called_this_episode, cumulative_reward
- Import Action, Observation, State from openenv.core.env_server
- Use @dataclass + field(default_factory=...)

## [envs/pr_review_env/server/tasks.py] — TO BUILD
### Purpose
Task bank loader. Reads JSONL of PR scenarios.
### Requirements
- @dataclass PRTask with fields: task_id, diff_str, pr_description, language, expected_verdict, primary_issue, difficulty
- load_tasks(path="tasks/tasks.jsonl") -> list[PRTask]
- get_random_task() -> PRTask
- If tasks.jsonl doesn't exist, return 5 hardcoded fallback tasks from skills/pr-tasks/SKILL.md
- Do NOT generate synthetic tasks here — that's a separate script

## [envs/pr_review_env/server/grader.py] — TO BUILD
### Purpose
Reward aggregation from tool outputs and terminal verdicts.
### Requirements
- aggregate_tool_scores(tool_results: dict) -> float (weighted mean, weights security=0.4 linter=0.3 complexity=0.3, renormalize if tools missing)
- step_reward(tool_name: str, already_called: bool, step_count: int) -> float
  - +0.10 for new informative tool call
  - -0.20 for calling same tool twice
  - -0.02 * max(0, step_count - 3) step penalty
- terminal_reward(submitted_verdict: str, expected_verdict: str, tool_results: dict) -> float
  - +1.0 for correct verdict with supporting tool evidence
  - +0.3 for correct verdict with no evidence
  - -0.5 for wrong verdict
- Write pytest tests: tests/test_grader.py

## [envs/pr_review_env/server/tools.py] — TO BUILD
### Purpose
FastMCP tool definitions wrapping semgrep/pylint/radon subprocess calls.
### Requirements
- Create FastMCP("pr-review-tools") instance named mcp (exported at module level)
- @mcp.tool call_security(diff_str: str) -> dict (runs semgrep --config=auto)
- @mcp.tool call_linter(diff_str: str) -> dict (runs pylint)
- @mcp.tool call_complexity(diff_str: str) -> dict (runs radon cc --json)
- @mcp.tool submit_review(verdict: str, reasoning: str) -> dict (terminal)
- @mcp.tool escalate(reason: str) -> dict (terminal)
- All subprocess calls with 30s timeout, temp file cleanup in finally blocks
- NEVER use tool names: reset, step, state, close

## [envs/pr_review_env/server/pr_review_env.py] — TO BUILD
### Purpose
MCPEnvironment subclass that glues tools + grader + tasks together.
### Requirements
- Import mcp from .tools, compute_step_reward etc from .grader, get_random_task from .tasks
- Class PRReviewEnv(MCPEnvironment)
- __init__(self): super().__init__(mcp_server=mcp); init state tracking
- reset() -> PRReviewObservation
- _step_impl(action) -> PRReviewObservation (non-MCP fallback, minimal)
- The CallToolAction routing is handled by MCPEnvironment automatically;
  override step() only to inject reward calculation after each tool call

## [envs/pr_review_env/server/app.py] — TO BUILD
### Purpose
FastAPI entrypoint.
### Requirements
- from openenv.core.env_server import create_fastapi_app
- Instantiate PRReviewEnv
- app = create_fastapi_app(env, PRReviewAction, PRReviewObservation)

## [envs/pr_review_env/server/Dockerfile] — TO BUILD
### Purpose
Container build for HF Spaces.
### Requirements
- Follow envs/README.md Dockerfile pattern exactly
- Base image: openenv-base:latest
- Copy requirements.txt, pip install, then copy env code
- Port 8000, healthcheck via /health
- CMD: uvicorn envs.pr_review_env.server.app:app --host 0.0.0.0 --port 8000

## [envs/pr_review_env/server/requirements.txt] — TO BUILD
### Purpose
Server-side Python deps.
### Requirements
Contents:
openenv-core
fastmcp
semgrep
radon
pylint
fastapi
uvicorn

## [envs/pr_review_env/client/pr_review_env_client.py] — TO BUILD
### Purpose
Typed HTTP client for training scripts.
### Requirements
- Class PRReviewEnv(HTTPEnvClient) with ACTION_CLASS, OBSERVATION_CLASS, STATE_CLASS
- _parse_result and _parse_state methods
