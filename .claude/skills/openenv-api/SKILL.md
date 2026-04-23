# OpenEnv API Skill

READ THIS BEFORE WRITING ANY ENVIRONMENT CODE.
This supersedes any earlier blueprint assumptions.

## OpenEnv is a client-server framework, NOT a gym wrapper

An OpenEnv environment is:
- A FastAPI HTTP server (with optional MCP JSON-RPC endpoint)
- Running inside a Docker container
- Deployed to Hugging Face Spaces via `openenv push`
- Consumed by clients (trainers, agents) over HTTP/WebSocket

## Directory layout (MANDATORY)

envs/pr_review_env/
├── models.py                        # Pydantic Action/Observation/State
├── client/
│   └── pr_review_env_client.py      # HTTPEnvClient subclass (typed)
└── server/
    ├── app.py                       # create_fastapi_app entrypoint
    ├── pr_review_env.py             # MCPEnvironment subclass
    ├── tools.py                     # FastMCP server with @mcp.tool functions
    ├── grader.py                    # Reward aggregation
    ├── tasks.py                     # PR task loader
    ├── requirements.txt
    └── Dockerfile

## models.py — action/observation/state dataclasses

```python
from dataclasses import dataclass, field
from openenv.core.env_server import Action, Observation, State

@dataclass
class PRReviewAction(Action):
    """Agent either calls an MCP tool or submits a final verdict."""
    tool_name: str = ""        # one of: call_security, call_linter, call_complexity, submit_review, escalate
    arguments: dict = field(default_factory=dict)

@dataclass
class PRReviewObservation(Observation):
    diff_str: str = ""          # current PR diff being reviewed
    pr_description: str = ""
    tools_called: list = field(default_factory=list)    # e.g. ["call_security"]
    tool_results: dict = field(default_factory=dict)    # tool_name -> result dict
    step_count: int = 0
    done: bool = False
    reward: float = 0.0

@dataclass
class PRReviewState(State):
    task_id: str = ""
    expected_verdict: str = ""
    tools_called_this_episode: list = field(default_factory=list)
    cumulative_reward: float = 0.0
```

## server/tools.py — FastMCP tool definitions

```python
from fastmcp import FastMCP
import subprocess, json, tempfile, os

mcp = FastMCP("pr-review-tools")

@mcp.tool
def call_security(diff_str: str) -> dict:
    """Run semgrep on the provided diff. Returns findings + score 0-1."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(diff_str)
        path = f.name
    try:
        result = subprocess.run(
            ["semgrep", "--config=auto", path, "--json", "--quiet"],
            capture_output=True, text=True, timeout=30,
        )
        data = json.loads(result.stdout or "{}")
        findings = data.get("results", [])
        score = max(0.0, 1.0 - 0.15 * sum(1 for f in findings if f.get("extra", {}).get("severity") == "ERROR"))
        return {"tool": "security", "score": score, "findings": findings[:10]}
    finally:
        os.unlink(path)

@mcp.tool
def call_linter(diff_str: str) -> dict:
    """Run pylint on the diff."""
    # similar pattern
    ...

@mcp.tool
def call_complexity(diff_str: str) -> dict:
    """Run radon cyclomatic complexity analysis."""
    ...

@mcp.tool
def submit_review(verdict: str, reasoning: str) -> dict:
    """Submit final verdict: approve, request_changes, or reject."""
    return {"verdict": verdict, "reasoning": reasoning, "terminal": True}

@mcp.tool
def escalate(reason: str) -> dict:
    """Escalate to human reviewer."""
    return {"escalated": True, "reason": reason, "terminal": True}
```

**RESERVED NAMES — DO NOT USE:** reset, step, state, close. These will raise ValueError.

## server/pr_review_env.py — MCPEnvironment subclass

```python
from openenv.core.env_server import MCPEnvironment
from .tools import mcp
from .tasks import get_random_task
from .grader import compute_episode_reward

class PRReviewEnv(MCPEnvironment):
    def __init__(self):
        super().__init__(mcp_server=mcp)
        self._current_task = None
        self._tools_called = []
        self._tool_results = {}
        self._step_count = 0

    def reset(self):
        self._current_task = get_random_task()
        self._tools_called = []
        self._tool_results = {}
        self._step_count = 0
        return PRReviewObservation(
            diff_str=self._current_task.diff_str,
            pr_description=self._current_task.pr_description,
            step_count=0,
        )

    def _step_impl(self, action):
        # Called for non-MCP actions (should be rare since all actions ARE tools)
        pass

    # MCPEnvironment handles CallToolAction routing to FastMCP tools automatically
```

## server/app.py — FastAPI entry point

```python
from openenv.core.env_server import create_fastapi_app
from .pr_review_env import PRReviewEnv
from ..models import PRReviewAction, PRReviewObservation

env = PRReviewEnv()
app = create_fastapi_app(env, PRReviewAction, PRReviewObservation)
```

## server/Dockerfile

```dockerfile
ARG BASE_IMAGE=openenv-base:latest
FROM ${BASE_IMAGE}
COPY envs/pr_review_env/server/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt
COPY envs/pr_review_env/ /app/envs/pr_review_env/
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1
CMD ["uvicorn", "envs.pr_review_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

## server/requirements.txt

openenv-core
fastmcp
semgrep
radon
pylint
fastapi
uvicorn

## client/pr_review_env_client.py — typed client

```python
from openenv.core.env_client import HTTPEnvClient
from ..models import PRReviewAction, PRReviewObservation, PRReviewState

class PRReviewEnv(HTTPEnvClient):
    ACTION_CLASS = PRReviewAction
    OBSERVATION_CLASS = PRReviewObservation
    STATE_CLASS = PRReviewState

    def _parse_result(self, data):
        return PRReviewObservation(**data)

    def _parse_state(self, data):
        return PRReviewState(**data)
```

## CLI commands (the real ones)

- `openenv init <name>` — scaffold new environment from template
- `openenv validate <path>` — check environment is correctly structured
- `openenv build <path>` — build Docker image locally
- `openenv serve <path>` — run environment locally
- `openenv push --repo-id user/env-name` — deploy to HF Spaces
- `openenv fork <hf-space>` — duplicate an existing environment

There is NO `openenv deploy` command.

## Local testing

```bash
# Fast iteration (no Docker)
cd envs/pr_review_env/server
uvicorn app:app --reload --port 8000

# Full Docker build
openenv build envs/pr_review_env/

# Push to Hub
openenv push --repo-id YOUR_USERNAME/pr-review-env
```

## Client usage pattern

```python
import asyncio
from envs.pr_review_env.client.pr_review_env_client import PRReviewEnv
from openenv.core.env_client import CallToolAction

async def main():
    async with PRReviewEnv(base_url="http://localhost:8000") as env:
        obs = await env.reset()
        result = await env.step(CallToolAction(
            tool_name="call_security",
            arguments={"diff_str": obs.observation.diff_str},
        ))
        print(result.reward, result.observation)

asyncio.run(main())
```
EOF

