# Tasks and Review Process

The default benchmark is `tasks/all_tasks.jsonl`. It contains `78` tasks:

- `65` seed tasks
- `13` comprehensive multi-file tasks

Each task is a typed `PRTask`:

```python
PRTask(
    task_id="py_sql_injection",
    diff_str="...",
    pr_description="Adds a Flask search endpoint for customer records.",
    primary_language="python",
    changed_file_types=["python"],
    repo_kind="backend_service",
    expected_verdict="reject",
    risk_domains=["security"],
    difficulty="easy",
    review_goal="Catch obvious injection and avoid approving without evidence.",
)
```

## Task Bank Interfaces

Every training and evaluation entrypoint accepts the same task interface:

```bash
--task-bank all
--task-bank seed
--task-bank comprehensive
--tasks-file path/to/custom.jsonl
```

The loader mode is also shared:

```bash
--task-loader-mode short
```

`short` is the default for training. It gives the prompt enough organizational
context to matter without making every GRPO sample expensive.

## Example 1: SQL Injection

Task: `py_sql_injection`

Diff:

```diff
+from flask import request
+
+def search_users(conn):
+    query = request.args.get("q", "")
+    sql = f"SELECT * FROM users WHERE name LIKE '%{query}%'"
+    return conn.execute(sql).fetchall()
```

Metadata:

```json
{
  "primary_language": "python",
  "changed_file_types": ["python"],
  "risk_domains": ["security"],
  "expected_verdict": "reject"
}
```

Good review process:

1. call `check_security`
2. receive a finding about string-built SQL from request input
3. submit `reject`

Example final action:

```json
{
  "tool_name": "submit_review",
  "arguments": {
    "verdict": "reject",
    "confidence": 0.9,
    "reasoning": "User input is interpolated into SQL before execution."
  }
}
```

## Example 2: JWT Validation Removed

Task: `comp_auth_jwt_removed`

This is a comprehensive task. It spans middleware, model schema, and tests. The
risk is not just one line: the PR removes signature and expiry checks, removes
`exp` from the token payload model, and deletes the expired-token test.

Diff excerpt:

```diff
-        try:
-            payload = jwt.decode(token, self.secret_key, algorithms=["RS256"],
-                                 options={"verify_signature": True, "verify_exp": True})
-        except jwt.ExpiredSignatureError:
-            return JSONResponse({"error": "Token expired"}, status_code=401)
-        except jwt.InvalidTokenError:
-            return JSONResponse({"error": "Invalid token"}, status_code=401)
+        payload = jwt.decode(token, options={"verify_signature": False})
```

Expected verdict: `reject`

Good route:

```json
{"tool_name": "check_security", "arguments": {}}
```

Optionally:

```json
{"tool_name": "check_tests", "arguments": {}}
```

Then submit:

```json
{
  "tool_name": "submit_review",
  "arguments": {
    "verdict": "reject",
    "confidence": 0.95,
    "reasoning": "JWT signature and expiry validation were removed, and the regression test was deleted."
  }
}
```

## Example 3: Infra Secrets and Workflow Permissions

Task: `comp_infra_secrets_exposed`

This task is mixed config and CI/CD:

- Docker build args contain secrets
- secrets are promoted to image environment variables
- `USER nonroot` is removed
- GitHub Actions permissions are broad

Diff excerpt:

```diff
+ARG DATABASE_URL
+ARG SECRET_KEY
+ARG STRIPE_SECRET_KEY
+
+ENV DATABASE_URL=${DATABASE_URL}
+ENV SECRET_KEY=${SECRET_KEY}
+ENV STRIPE_SECRET_KEY=${STRIPE_SECRET_KEY}
...
-USER nonroot
+RUN useradd -m appuser
```

Expected verdict: `reject`

Good route:

```json
{"tool_name": "check_config", "arguments": {}}
```

Then:

```json
{"tool_name": "check_security", "arguments": {}}
```

The important behavior is that the model must not treat this as a normal Python
backend PR just because the repo is Python. The risky files are Dockerfile and
GitHub Actions.

## Example 4: Clean Refactor

Task: `go_clean_refactor`

Diff:

```diff
+func validateID(id string) error {
+    if id == "" {
+        return errors.New("missing id")
+    }
+    return nil
+}
```

Expected verdict: `approve`

A good policy should avoid over-reviewing this. One light quality or build check
is enough; repeated security/config checks waste steps and can reduce reward.

Example final action:

```json
{
  "tool_name": "submit_review",
  "arguments": {
    "verdict": "approve",
    "confidence": 0.8,
    "reasoning": "Small validation helper with no risky behavior detected."
  }
}
```

## What Happens Internally

When the model runs on a task:

1. `PRReviewEnv.reset(...)` loads the task.
2. `task_review_config(...)` builds short loader context from `docs/`.
3. `build_obs_prompt(...)` formats the observation.
4. the model emits one JSON action.
5. `_action_with_state_args(...)` fills missing `diff_str`, `task_id`, and
   default terminal arguments.
6. the environment executes the tool.
7. `step_reward(...)` scores the tool choice.
8. the observation records tool history.
9. the model continues until `submit_review` or `escalate`.
10. `terminal_reward(...)` and `evidence_penalties(...)` score the final result.

This structure makes every review auditable. A result is not just a verdict; it
is a sequence of tool calls and evidence.
