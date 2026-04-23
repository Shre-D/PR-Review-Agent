# FastMCP Tool Skill

Reference when writing tools for MCPEnvironment.

## Basic pattern

```python
from fastmcp import FastMCP

mcp = FastMCP("server-name")

@mcp.tool
def my_tool(arg1: str, arg2: int = 10) -> dict:
    """Docstring becomes the tool description shown to the agent."""
    return {"result": "..."}
```

## Type support

- Tool inputs: any Pydantic-compatible type (primitives, lists, dicts, Pydantic models)
- Tool outputs: dict (preferred), str, primitives, None, or Pydantic models

## Async tools

```python
@mcp.tool
async def fetch_thing(url: str) -> dict:
    import httpx
    async with httpx.AsyncClient() as c:
        resp = await c.get(url)
        return resp.json()
```

## Error handling

```python
from fastmcp import ToolError

@mcp.tool
def risky(arg: str) -> dict:
    if not arg:
        raise ToolError("arg must be non-empty")   # visible to agent
    try:
        ...
    except Exception:
        raise  # masked from agent, appears as generic error
```

## Reserved tool names (will raise ValueError when registered with MCPEnvironment)

- `reset`
- `step`
- `state`
- `close`

Pick anything else. Common safe names: `call_*`, `submit_*`, `fetch_*`, `run_*`.

## For this project, your tools are

- `call_security` — runs semgrep
- `call_linter` — runs pylint
- `call_complexity` — runs radon cc
- `submit_review` — terminal action with verdict
- `escalate` — terminal action for "needs human"

## Running semgrep/radon/pylint as subprocesses

Use `subprocess.run(..., capture_output=True, text=True, timeout=N)`.
Always set a timeout (10-30s). Always wrap in try/finally if writing temp files.
Parse `--json` output, don't regex stdout.
