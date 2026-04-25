# Final Submission Spec

## Goal

Build a working OpenEnv-compatible benchmark where a small policy model reviews
PR diffs by choosing the right tool calls before emitting a final verdict.

## Submission Scope

### Code languages

- Python
- TypeScript/JavaScript
- Java
- Go
- Rust

### Repo-global file types

- Dockerfile
- YAML
- GitHub Actions workflow YAML
- `.gitignore`

### Task families

- pure code PRs
- pure config / infra PRs
- mixed PRs

## Runtime Tools

- `check_security`
- `check_quality`
- `check_build_and_types`
- `check_tests`
- `check_config`
- `submit_review`
- `escalate`

## Environment API

### `envs/pr_review_env/models.py`

- `PRReviewAction`: one tool call per step
- `PRReviewObservation`: diff, task metadata, tool history, last tool result
- `PRReviewState`: internal episode state for scoring and baselines

#### Detailed Spec (for codex-writer)

File: `envs/pr_review_env/models.py`
Purpose: Pydantic dataclasses for the OpenEnv action/observation/state protocol.
Imports from: `.compat` (CallToolAction, Observation, State — real OpenEnv or fallback)

```
Verdict = Literal["approve", "request_changes", "reject", "escalate"]

class PRReviewAction(CallToolAction):
    """One environment action equals one tool invocation."""
    # No extra fields — inherits tool_name: str and arguments: dict from CallToolAction

class PRReviewObservation(Observation):
    # Inherited from Observation: done: bool = False, reward: float | None = None, metadata: dict
    diff_str: str = ""                              # raw unified diff of the PR
    pr_description: str = ""                        # PR title + body text
    primary_language: str = ""                      # dominant language (python, go, rust, …)
    changed_file_types: list[str] = []              # e.g. ["py", "yaml", "Dockerfile"]
    repo_kind: str = ""                             # "library", "service", "infra", "mixed"
    available_tools: list[str] = []                 # tools the agent may call this episode
    tools_called: list[str] = []                    # ordered list of tool names called so far
    tool_results: dict[str, dict[str, Any]] = {}    # tool_name -> result dict
    review_history: list[str] = []                  # human-readable summary per step
    last_tool_name: str | None = None               # most recent tool invoked
    last_tool_result: dict[str, Any] = {}           # result from most recent tool call
    task_id: str = ""                               # unique task identifier
    step_count: int = 0                             # steps taken this episode

class PRReviewState(State):
    # Inherited from State: episode_id: str | None, step_count: int
    task_id: str = ""
    expected_verdict: str = ""                      # ground-truth verdict from task bank
    primary_language: str = ""
    changed_file_types: list[str] = []
    risk_domains: list[str] = []                    # e.g. ["security", "config", "tests"]
    tools_called_this_episode: list[str] = []
    tool_results: dict[str, dict[str, Any]] = {}
    review_history: list[str] = []
    cumulative_reward: float = 0.0
```

Constraints:
- Use `from __future__ import annotations` at top
- Use `pydantic.Field(default_factory=list/dict)` for all mutable defaults
- Do NOT use `@dataclass` — these must be Pydantic BaseModel subclasses
- No `model_config` overrides needed (inherit from base classes in compat.py)
- File must be importable with `python -c "from envs.pr_review_env.models import PRReviewAction, PRReviewObservation, PRReviewState; print('ok')"`

### `envs/pr_review_env/server/tasks.py`

- multi-language task schema
- JSONL task loader
- seeded fallback task bank

#### Detailed Spec (for codex-writer)

File: `envs/pr_review_env/server/tasks.py`
Purpose: Task schema, JSONL loader, and seeded fallback bank for the PR review benchmark.
No imports from models.py — pure stdlib dataclass.

```
TASKS_PATH = Path("tasks/tasks.jsonl")  # relative to repo root; resolved at call time

@dataclass
class PRTask:
    task_id: str                          # unique slug, e.g. "py_sql_injection"
    diff_str: str                         # unified diff text
    pr_description: str                   # PR title + body
    primary_language: str                 # "python", "go", "rust", "typescript", "java", "yaml", "dockerfile"
    changed_file_types: list[str]         # extensions present, e.g. ["py", "yaml"]
    repo_kind: str                        # "library" | "service" | "infra" | "mixed"
    expected_verdict: str                 # "approve" | "request_changes" | "reject"
    risk_domains: list[str]               # subset of ["security","quality","tests","config","build"]
    difficulty: str                       # "easy" | "medium" | "hard"
    review_goal: str                      # one-sentence description of what a reviewer should catch
    ownership_hint: str = ""              # optional team/owner hint
    notes: list[str] = field(default_factory=list)  # optional free-form notes

    def to_dict(self) -> dict:
        return asdict(self)

def _fallback_tasks() -> list[PRTask]:
    """12 hardcoded representative tasks covering all supported languages and file types."""
    # Must cover: py_sql_injection, ts_eval_template, java_null_contract, go_clean_refactor,
    #             rust_unwrap_io, docker_root_user, gha_permissions_write, gitignore_secret_file,
    #             yaml_prod_debug, mixed_python_workflow, ts_missing_test, java_dependency_bump
    ...

def load_tasks(path: str | Path = TASKS_PATH) -> list[PRTask]:
    """Load from JSONL; fall back to _fallback_tasks() if file missing or empty."""
    ...

def get_random_task(path: str | Path = TASKS_PATH, rng: random.Random | None = None) -> PRTask:
    """Return a random PRTask from the loaded pool."""
    ...

def get_task_by_id(task_id: str, path: str | Path = TASKS_PATH) -> PRTask:
    """Return the task with matching task_id; raise KeyError if not found."""
    ...

def dump_tasks(tasks: Iterable[PRTask], path: str | Path = TASKS_PATH) -> None:
    """Serialize tasks to JSONL, creating parent dirs as needed."""
    ...
```

Constraints:
- Pure stdlib + dataclasses — no pydantic, no external deps
- `from __future__ import annotations`
- `field(default_factory=list)` for list fields
- `TASKS_PATH` is relative; `load_tasks` must resolve it against `Path(__file__).parent.parent.parent` so it works regardless of cwd
- File must be importable: `python -c "from envs.pr_review_env.server.tasks import PRTask, get_random_task, load_tasks; print('ok')"`

### `envs/pr_review_env/server/tools.py`

- heuristic-first review tools
- optional hybrid use of semgrep, pylint, and radon
- config-aware checks for Dockerfile / YAML / workflow / `.gitignore`

#### Detailed Spec (for codex-writer)

File: `envs/pr_review_env/server/tools.py`
Purpose: FastMCP tool server with 7 tools: check_security, check_quality, check_build_and_types, check_tests, check_config, submit_review, escalate.
The file already exists at 646 lines. Apply targeted fixes only — do not rewrite the whole file.

**Fix 1 — `_heuristic_findings` gitignore heuristic (config section)**

Current (wrong):
```python
if ".env.production" in lower or "secrets/" in lower:
    findings["config"].append("Ignore rules may hide deployment secrets or critical config.")
```

Replace with (correct — detects narrowing of broad env-file protections via removed lines):
```python
removed_lines = [
    line[1:].strip()
    for line in diff_str.splitlines()
    if line.startswith("-") and not line.startswith("---")
]
if ".gitignore" in lower and any(
    ".env" in rl or "secret" in rl or "*.pem" in rl or "*.key" in rl
    for rl in removed_lines
):
    findings["config"].append("Gitignore narrowing removes broad env/secret protection — files may become trackable.")
elif ".env.production" in lower or "secrets/" in lower:
    findings["config"].append("Ignore rules may hide deployment secrets or critical config.")
```

**Fix 2 — `security_patterns` "Command execution" entry**

Current (wrong — requires ALL three unrelated patterns simultaneously):
```python
"Command execution surfaced in diff": ["runtime.getruntime().exec(", ".arg(\"-c\")", "pickle.loads("],
```

Replace with three separate single-pattern entries (each triggers independently):
```python
"Unsafe Java runtime exec detected": ["runtime.getruntime().exec("],
"Shell command injection risk (.arg -c pattern)": [".arg(\"-c\")"],
"Unsafe pickle deserialization": ["pickle.loads("],
```

After applying fixes, verify:
```bash
cd /home/shred/Desktop/Programming/Hackathon/PR-Review-Agent && python -c "
from envs.pr_review_env.server.tools import _heuristic_findings

# Test 1: gitignore narrowing (removed .env*) should trigger config finding
gitignore_diff = '''diff --git a/.gitignore b/.gitignore
index 1111111..2222222 100644
--- a/.gitignore
+++ b/.gitignore
@@
-.env*
+.env.local
+.env.development
'''
f = _heuristic_findings(gitignore_diff)
assert f['config'], f'gitignore narrowing not detected: {f[\"config\"]}'

# Test 2: pickle.loads alone should trigger security finding
pickle_diff = '+result = pickle.loads(data)'
f2 = _heuristic_findings(pickle_diff)
assert any('pickle' in x.lower() for x in f2['security']), f'pickle not detected: {f2[\"security\"]}'

print('ok')
"
```

### `envs/pr_review_env/server/grader.py`

- reward for relevant new tool calls
- penalty for duplicates
- step penalty after the efficient review window
- terminal reward based on verdict correctness plus evidence quality

#### Detailed Spec (for codex-writer)

File: `envs/pr_review_env/server/grader.py`
Purpose: Stateless reward functions for the PR review benchmark.
Imports: only `from __future__ import annotations`, `from typing import Any`, `from .tasks import PRTask`

```python
# Valid risk_domains: {"security", "quality", "tests", "config", "build"}
# Valid tool names:   {"check_security", "check_quality", "check_build_and_types", "check_tests", "check_config"}

TOOL_WEIGHTS = {
    "check_security":       0.28,
    "check_quality":        0.20,
    "check_build_and_types":0.18,
    "check_tests":          0.16,
    "check_config":         0.18,
}

# Maps each valid risk_domain to the tools that cover it
DOMAIN_TO_TOOLS: dict[str, set[str]] = {
    "security": {"check_security"},
    "quality":  {"check_quality"},
    "tests":    {"check_tests"},
    "config":   {"check_config"},
    "build":    {"check_build_and_types", "check_config"},
}

def _tool_score(result: dict[str, Any] | None) -> float:
    """Extract normalised 0-1 score from a tool result dict."""
    # returns result["score"] clamped to [0, 1]; 0.0 if missing/invalid

def relevant_tools(task: PRTask) -> set[str]:
    """Return the set of tool names relevant to this task's risk_domains.

    If risk_domains is empty (clean diff), return {"check_quality"} as a
    minimal baseline — the agent should still do a sanity check.
    Otherwise union DOMAIN_TO_TOOLS[d] for each d in task.risk_domains.
    Do NOT reference "ci" or "clean" — those are not valid domains.
    """

def aggregate_tool_scores(tool_results: dict[str, dict[str, Any]]) -> float:
    """Weighted average of _tool_score() for tools present in TOOL_WEIGHTS.
    Weights are renormalised over only the tools that were actually called.
    Returns 0.0 if no known tools were called. Rounded to 3 decimal places."""

def step_reward(
    task: PRTask,
    tool_name: str,
    already_called: bool,
    step_count: int,
) -> float:
    """Per-step reward for calling a tool (not submit_review / escalate).

    Rules:
    - submit_review / escalate → return 0.0 (terminal, handled separately)
    - already_called (duplicate) → -0.20
    - new call: base +0.05; +0.08 bonus if tool_name in relevant_tools(task)
    - efficiency penalty: -0.03 per step beyond step 4
    Rounded to 3 decimal places."""

def terminal_reward(
    task: PRTask,
    submitted_verdict: str,
    tool_results: dict[str, dict[str, Any]],
) -> float:
    """Reward on submit_review or escalate.

    Correct verdict + ≥1 supportive tool called → 1.0 + min(0.25, evidence*0.25)
    Correct verdict + no supportive tools        → 0.35
    "escalate" when expected reject/request_changes → 0.10  (partial credit)
    Wrong verdict                                → -0.55
    """

def outcome_summary(
    task: PRTask,
    tool_results: dict[str, dict[str, Any]],
    submitted_verdict: str | None = None,
) -> dict[str, Any]:
    """Return a dict with: expected_verdict, relevant_tools (sorted list),
    aggregate_score, submitted_verdict."""
```

Constraints:
- Pure Python — no external deps
- `relevant_tools()` must NOT reference `"ci"` or `"clean"` — only the 5 valid domains
- File must import cleanly: `python -c "from envs.pr_review_env.server.grader import terminal_reward, step_reward, relevant_tools; print('ok')"`

### `envs/pr_review_env/server/pr_review_env.py`

- environment orchestration
- episode state tracking
- conversion from raw tool outputs into review observations

#### Detailed Spec (for codex-writer)

File: `envs/pr_review_env/server/pr_review_env.py`
Purpose: MCPEnvironment subclass orchestrating one PR review episode.
Key imports: compat (CallToolObservation, MCPEnvironment, Observation, ensure_episode_id), models, grader, tasks, tools.

```python
class PRReviewEnv(MCPEnvironment):
    def __init__(self, task_path="tasks/tasks.jsonl", seed=None)
    def reset(self, seed=None, episode_id=None, task_id=None, **kwargs) -> PRReviewObservation
    @property def state(self) -> PRReviewState
    def step(self, action: PRReviewAction, timeout_s=None, **kwargs) -> PRReviewObservation
    def _extract_result_payload(self, observation: CallToolObservation) -> dict
    def _observation(self, reward, last_tool_name, last_tool_result) -> PRReviewObservation
    def _step_impl(self, action, ...) -> raises TypeError (all actions are tool calls)
```

### `envs/pr_review_env/server/app.py`

- FastAPI entrypoint — passes PRReviewEnv class + action/observation types to create_fastapi_app

#### Detailed Spec (for codex-writer)

File: `envs/pr_review_env/server/app.py`
Purpose: ASGI entrypoint. Exactly 3 imports + one assignment.

```python
from __future__ import annotations

from ..compat import create_fastapi_app
from ..models import PRReviewAction, PRReviewObservation
from .pr_review_env import PRReviewEnv

app = create_fastapi_app(PRReviewEnv, PRReviewAction, PRReviewObservation)
```

Constraints:
- `PRReviewEnv` passed as a class (callable factory), not an instance
- No other symbols defined — uvicorn targets `envs.pr_review_env.server.app:app`
- Must import cleanly: `python -c "from envs.pr_review_env.server.app import app; print('ok', app.title)"`

### `envs/pr_review_env/server/Dockerfile`

- ARG BASE_IMAGE=openenv-base:latest
- Installs requirements.txt, copies env + tasks dir, exposes 8000, healthcheck, CMD uvicorn

#### Detailed Spec (for codex-writer)

File: `envs/pr_review_env/server/Dockerfile`
File: `envs/pr_review_env/server/requirements.txt`
File: `pyproject.toml` (server optional-dependencies section only)

**requirements.txt** must list (flat, no version pins — runtime flexibility):
```
openenv-core
fastmcp
fastapi
uvicorn
httpx
pyyaml
semgrep
pylint
radon
pydantic
```
Key fix: `pyyaml` is MISSING — `tools.py` does `import yaml` which requires it.

**Dockerfile** — keep existing structure, one fix:
Replace `curl`-based HEALTHCHECK (curl may not be in base image) with Python stdlib:
```dockerfile
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1
```

**pyproject.toml** — add `pyyaml` to the `server` optional-dependencies list only.

Verify:
- requirements.txt contains `pyyaml`
- Dockerfile HEALTHCHECK does not use curl
- `python -c "import yaml; print('ok')"` passes (yaml already installed in dev env)

### `envs/pr_review_env/client/pr_review_env_client.py`

- Typed async HTTP client for the PR review environment server

#### Detailed Spec (for codex-writer)

File: `envs/pr_review_env/client/pr_review_env_client.py`
Purpose: HTTPEnvClient subclass with typed action/observation/state generics.
Imports: `from ..compat import EnvClient, StepResult` and `from ..models import PRReviewAction, PRReviewObservation, PRReviewState`

```python
class PRReviewEnvClient(EnvClient[PRReviewAction, PRReviewObservation, PRReviewState]):

    def _step_payload(self, action: PRReviewAction) -> dict:
        """Serialize action for the /step POST body."""
        return action.model_dump()

    def _parse_result(self, payload: dict) -> StepResult[PRReviewObservation]:
        """Parse /reset and /step responses.
        Handles both envelope format {"observation": {...}, "reward": ..., "done": ...}
        and flat format where the whole payload is the observation.
        Sets observation.reward and observation.done from the top-level fields.
        """

    def _parse_state(self, payload: dict) -> PRReviewState:
        """Parse /state response via model_validate."""
```

Constraints:
- No `model_config` overrides — inherit from `EnvClient`
- `__init__`, `reset`, `step`, `state` are inherited — do not override them
- `__all__ = ["PRReviewEnvClient"]` must be in `client/__init__.py`
- File must import cleanly: `python -c "from envs.pr_review_env.client.pr_review_env_client import PRReviewEnvClient; print('ok')"`

### `train/grpo_train.py`

- GRPO training loop using TRL + the PR review environment server
- Qwen/Qwen3-1.7B policy model, heuristic rollouts, per-step reward shaping

#### Detailed Spec (for codex-writer)

File: `train/grpo_train.py`
Purpose: Standalone GRPO training script. Connects to a running pr-review-env server, rolls out episodes with a small model, collects rewards, trains with TRL GRPOTrainer.

```
ROOT = Path(__file__).resolve().parents[1]   # repo root for sys.path

SYSTEM_PROMPT: str   # terse system prompt telling the model to output JSON tool calls

MODEL_NAME = "Qwen/Qwen3-1.7B"
ENV_URL = os.getenv("PR_REVIEW_ENV_URL", "http://localhost:8000")
MAX_STEPS_PER_EPISODE = 8
GRPO_CONFIG = GRPOConfig(
    output_dir="./grpo_checkpoint",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    num_generations=8,
    max_new_tokens=128,
    logging_steps=1,
)

def parse_action(text: str) -> PRReviewAction:
    """Parse model output (JSON) into PRReviewAction.
    Accepts: {"tool_name": "...", "arguments": {...}}
    Falls back to submit_review(approve) on parse error."""

def build_obs_prompt(obs: PRReviewObservation) -> str:
    """Format observation as a user message for the chat template.
    Include diff, pr_description, primary_language, available_tools, review_history."""

async def rollout_episode(env_client, model, tokenizer, device) -> dict:
    """Run one full episode and return {"prompt": str, "response": str, "reward": float}.
    Uses greedy sampling (do_sample=False) to get a deterministic trajectory.
    Accumulates step rewards + terminal reward."""

async def collect_trajectories(model, tokenizer, device, n: int = 8) -> list[dict]:
    """Collect n episodes using PRReviewEnvClient, return list of trajectory dicts."""

def build_dataset(trajectories: list[dict]) -> Dataset:
    """Convert trajectory list to HuggingFace Dataset with columns: prompt, response, reward."""

def main():
    """
    1. Load model + tokenizer (Qwen3-1.7B, bfloat16, device_map=auto)
    2. Set tokenizer.pad_token = tokenizer.eos_token
    3. Collect initial trajectories
    4. Build dataset
    5. Instantiate GRPOTrainer(model, config, dataset, tokenizer)
    6. trainer.train()
    7. trainer.save_model()
    """
```

Constraints:
- `from __future__ import annotations` at top
- `ROOT` path setup + sys.path insert (same pattern as export_teacher_traces.py)
- Imports: `trl`, `transformers`, `datasets`, `torch`, `asyncio`, `json`, `os`, `argparse`
- Import from local: `from envs.pr_review_env.client.pr_review_env_client import PRReviewEnvClient`
  and `from envs.pr_review_env.models import PRReviewAction, PRReviewObservation`
- No Claude/Gemini/Codex API calls — Qwen3-1.7B only
- `trl` and `transformers` are imported inside `main()` or at top with `try/except ImportError`
  so the file stays importable even if trl is not installed in dev
- `if __name__ == "__main__": main()` at bottom
- File must import cleanly (no trl required at import time):
  `python -c "import train.grpo_train; print('ok')"`

### `train/run_baseline.py`

- Training companion script: runs random + heuristic baselines over the full task bank and prints a compact comparison table

#### Detailed Spec (for codex-writer)

File: `train/run_baseline.py`
Purpose: CLI to evaluate random and heuristic baselines across all tasks. Used before/after GRPO training to measure improvement. Thin wrapper over `benchmarks/evaluate_baselines.evaluate_policy`.

```python
"""Run random and heuristic baselines across the full task bank.

Usage:
    python train/run_baseline.py [--seed 7] [--output rewards/baseline_eval.json]

Prints a compact ASCII comparison table to stdout and writes full JSON to --output.
"""
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.evaluate_baselines import evaluate_policy


def print_summary(results: dict) -> None:
    """Print a compact comparison table:

    Policy      Tasks  Accuracy  Mean Return
    ----------  -----  --------  -----------
    random        65     0.231        0.123
    heuristic     65     0.615        0.891
    """

def main() -> None:
    parser = argparse.ArgumentParser(...)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default=str(ROOT / "rewards" / "baseline_eval.json"))
    args = parser.parse_args()

    results = {
        "heuristic": evaluate_policy("heuristic", seed=args.seed),
        "random": evaluate_policy("random", seed=args.seed),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print_summary(results)
    print(f"\nFull results written to: {output_path}")

if __name__ == "__main__":
    main()
```

Constraints:
- `from __future__ import annotations` at top
- Same `ROOT` / `sys.path` pattern as other train/ scripts
- Only imports from stdlib + `benchmarks.evaluate_baselines` (no new deps)
- `print_summary` must produce a human-readable table (no pandas — plain f-strings)
- File must import cleanly: `python -c "import train.run_baseline; print('ok')"`

## Baseline Story

- random router
- heuristic language-aware router
- learned small-model policy

## Non-Goals For Submission

- 15-language coverage
- true multi-agent software-team simulation
- full build/test execution for every ecosystem
- dependence on proprietary developer assistants during rollout
