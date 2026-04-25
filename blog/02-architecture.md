---
title: "Inside the OpenEnv PR Review Environment"
date: 2026-04-24
tags: [openenv, fastapi, mcp, architecture]
---

# Inside the OpenEnv PR Review Environment

The PR Review Environment is built on OpenEnv — a client-server framework for reinforcement learning environments. Unlike gym-style wrappers, OpenEnv environments are real HTTP servers running inside Docker containers. That design decision has consequences that ripple through the whole system.

## Why Client-Server?

A gym environment lives in the same process as your training loop. That works fine for toy environments, but breaks down for code review:

- Analysis tools (semgrep, pylint, cargo check) need their own binaries and dependencies
- Fixture-backed workspaces need filesystem access
- You want to scale training workers independently from environment workers
- You want to deploy the environment to HuggingFace Spaces without bundling your training infra

OpenEnv solves this by making the environment a FastAPI server. Training loops talk to it over HTTP. The environment can run on different hardware, be replicated, or be shared across multiple training runs.

## The Server Side

The environment server has four key components:

### `pr_review_env.py` — MCPEnvironment Subclass

```python
class PRReviewEnv(MCPEnvironment):
    def reset(self, task_id=None, seed=None, ...) -> PRReviewObservation:
        self._task = get_random_task() if not task_id else get_task_by_id(task_id)
        self._state = PRReviewState(task_id=..., expected_verdict=...)
        return self._observation(reward=0.0, ...)

    def step(self, action: PRReviewAction) -> PRReviewObservation:
        # Route to tool, compute reward, update state
        result = super().step(action)  # MCPEnvironment handles tool routing
        reward = step_reward(self._task, action.tool_name, ...)
        if action.tool_name in {"submit_review", "escalate"}:
            reward += terminal_reward(self._task, verdict, tool_results)
            self._done = True
        return self._observation(reward=reward, ...)
```

The `MCPEnvironment` base class handles the routing: when the agent calls `check_security`, it dispatches to the `check_security` function registered in the FastMCP server. The environment wraps the raw result with reward computation and state tracking.

### `tools.py` — FastMCP Tool Definitions

Each tool is a Python function decorated with `@mcp.tool`:

```python
@mcp.tool
def check_security(diff_str: str, task_id: str = "") -> dict:
    """Run security-focused review across supported code and config diffs."""
    heuristics = _heuristic_findings(diff_str)["security"]
    with _analysis_targets(diff_str, task_id=task_id) as targets:
        semgrep_payload, semgrep_findings = _semgrep_scan(targets)
        combined = list(heuristics) + semgrep_findings
        return {
            "score": _score_from_findings(combined, penalty=0.2),
            "findings": combined,
            ...
        }
```

Tools have three analysis modes:
- **fixture_backed**: A real workspace with the actual files from the task's fixture directory
- **diff_reconstructed**: A temporary file created from added lines in the diff
- **heuristic_only**: Pattern matching on the diff text alone (always available, no external deps)

### `app.py` — FastAPI Entrypoint

```python
from envs.pr_review_env.compat import create_fastapi_app
from .pr_review_env import PRReviewEnv

app = create_fastapi_app(PRReviewEnv, PRReviewAction, PRReviewObservation)
```

This creates `/reset`, `/step`, `/state`, and `/health` endpoints. That's the complete API surface the training loop needs.

## The Client Side

```python
class PRReviewEnvClient(EnvClient[PRReviewAction, PRReviewObservation, PRReviewState]):
    async def reset(self) -> StepResult[PRReviewObservation]: ...
    async def step(self, action: PRReviewAction) -> StepResult[PRReviewObservation]: ...
    async def state(self) -> PRReviewState: ...
```

The typed client wraps the HTTP calls and deserializes responses into Pydantic models. Training loops use it like this:

```python
async with PRReviewEnvClient(base_url="http://localhost:8000") as env:
    obs = await env.reset()
    result = await env.step(PRReviewAction(
        tool_name="check_security",
        arguments={"diff_str": obs.observation.diff_str}
    ))
    print(result.reward, result.observation.done)
```

## The Compatibility Layer

Since OpenEnv's upstream package pins are currently in flux, we ship a `compat.py` that provides drop-in fallbacks for all OpenEnv types. When the real package is available, it uses it. When it isn't, the fallback implementations keep everything importable and testable:

```python
try:
    from openenv.core.env_server import MCPEnvironment, Observation, State
    OPENENV_AVAILABLE = True
except Exception:
    OPENENV_AVAILABLE = False
    class MCPEnvironment: ...  # full fallback implementation
```

This design means the environment imports and runs correctly in any Python environment — no Docker, no OpenEnv package — just pure Python.

---

*Next: [Building a Multi-Language PR Benchmark](./03-task-bank.md)*
