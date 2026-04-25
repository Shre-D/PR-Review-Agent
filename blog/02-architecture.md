# Architecture

The project has four main layers:

1. task bank
2. OpenEnv-compatible environment
3. review tools and reward logic
4. training and evaluation scripts

The policy model sits outside the environment. It receives observations and
emits JSON tool calls. The environment owns task state, tool execution, reward
calculation, and terminal verdict scoring.

## Component Map

| Component | Path | Role |
|---|---|---|
| Typed models | `envs/pr_review_env/models.py` | Pydantic action, observation, state, verdict, and config models |
| Environment | `envs/pr_review_env/server/pr_review_env.py` | Reset/step loop, state tracking, reward integration |
| Tools | `envs/pr_review_env/server/tools.py` | Security, quality, build/type, test, config, submit, escalate tools |
| Grader | `envs/pr_review_env/server/grader.py` | Per-step reward, terminal reward, evidence penalties |
| Task loader | `envs/pr_review_env/server/tasks.py` | Task aliases, JSONL loading, combined bank, per-task loader config |
| Context loader | `envs/pr_review_env/server/context_loader.py` | Deterministic structural config extraction from docs |
| Training | `train/grpo_train.py` | GRPO/QLoRA training with online environment reward |
| Evaluation | `benchmarks/evaluate_trained_model.py` | Runs a LoRA checkpoint against a task bank |
| Baselines | `benchmarks/evaluate_baselines.py` | Random and heuristic policies |

## Episode Flow

An episode starts with `PRReviewEnv.reset(task_id=...)`.

The environment loads a `PRTask` and returns a `PRReviewObservation`:

```python
PRReviewObservation(
    diff_str="...",
    pr_description="...",
    primary_language="python",
    changed_file_types=["python"],
    available_tools=["check_security", "check_quality", ...],
    review_history=[],
    task_id="py_sql_injection",
)
```

The policy emits a `PRReviewAction`:

```json
{"tool_name": "check_security", "arguments": {}}
```

Before tool execution, the environment fills in state-aware arguments. For
analysis tools it adds the diff and task id. If loader context is enabled, it
also passes the current `review_config`.

The tool returns a structured result:

```json
{
  "tool": "check_security",
  "score": 0.64,
  "findings": ["Potential SQL injection via string-built query"],
  "backend": "heuristic",
  "analysis_mode": "fixture_backed"
}
```

The grader assigns a step reward, updates review history, and returns the next
observation. When the policy submits a verdict, the terminal reward compares the
verdict to the expected task label and checks whether the evidence supports it.

## The Loader Boundary

The context loader is deliberately not a runtime LLM dependency. It is a
deterministic structural parser over files in `docs/`. It can extract:

- architecture summary
- critical paths
- tool weights
- domain priorities
- author-depth overrides
- enabled/planned external tools

Training normally uses:

```bash
--task-loader-mode short
```

That gives each task a compact config: enough for the model to learn that
critical paths and organizational context matter, but short enough to keep GRPO
prompt cost manageable.

Other modes exist for controlled experiments:

- `empty`: exercise the loader path with almost no prompt text
- `full`: pass the full structural config
- `off`: disable loader config

## Why the Architecture Is Small

The system avoids a large orchestration stack. There is no separate queue,
database, vector store, or agent framework. The benchmark only needs to answer:

1. what task is being reviewed?
2. what tools did the policy call?
3. what evidence came back?
4. what reward should that action receive?
5. what final verdict was submitted?

Keeping that loop small makes the training path auditable. It also means the
same task loader and reward code are used by baselines, GRPO, trained-model
evaluation, the UI smoke path, HF Jobs, and HPC.

## Server and Client

`envs/pr_review_env/server/app.py` exposes the environment through the
OpenEnv-compatible FastAPI wrapper. The client in
`envs/pr_review_env/client/pr_review_env_client.py` parses reset/step responses
back into typed observations.

Training currently computes rewards locally by replaying task state inside
`PRReviewEnv`, so the GRPO loop does not need to run a remote server. The server
still matters for demos, integration tests, and compatibility with OpenEnv
workflows.
