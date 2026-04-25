# Reward System

The reward function teaches the policy that a review is not only about the
final verdict. It also cares about the evidence path used to reach that verdict.

A good episode should:

- call relevant tools
- avoid duplicate or irrelevant tools
- submit the correct verdict
- avoid approving when evidence shows critical findings
- avoid rejecting clean changes without support

## Step Reward

Every non-terminal tool call receives a step reward from
`envs/pr_review_env/server/grader.py`.

The main rules are:

- new relevant tool call: `+0.3` raw reward
- new irrelevant tool call: `+0.1` raw reward
- duplicate tool call: `−0.4` penalty
- too many steps for the task difficulty: `−0.05` per step past threshold

The gap between relevant (`+0.3`) and irrelevant (`+0.1`) is `0.2` — large
enough for GRPO to compute meaningful advantages across completions.

Example: a security task such as `py_sql_injection` has `risk_domains` containing
`security`, so `check_security` is relevant and earns `+0.3`. A first call to
`check_config` would only earn `+0.1`.

For clean tasks, the relevant baseline tool is `check_quality`. This prevents
the model from learning that every PR must go through the security path.

## Relevant Tools

The default risk-domain mapping is:

| Risk Domain | Relevant Tools |
|---|---|
| `security` | `check_security` |
| `quality` | `check_quality` |
| `tests` | `check_tests` |
| `config` | `check_config` |
| `build` | `check_build_and_types`, `check_config` |

The context loader can adjust priorities, but the action space stays stable.
The policy always chooses from:

- `check_security`
- `check_quality`
- `check_build_and_types`
- `check_tests`
- `check_config`
- `submit_review`
- `escalate`

## Terminal Reward

Terminal reward is assigned when the model calls `submit_review` or `escalate`.
These are raw values returned directly to GRPO:

- correct verdict with supportive evidence: `+1.0` to `+1.4`
- correct verdict, early submit (insufficient evidence): `+0.2`
- correct verdict, no supportive tool: `−0.3`
- escalation on risky tasks: `−0.2` to `−0.4`
- wrong verdict: `−0.8` (with evidence) / `−1.0` (early)

The gap between evidence-backed correct (`+1.0`) and early correct (`+0.2`) is
`+0.8`. This is the most important signal: the model must learn that correct
verdicts without evidence are worth far less than ones backed by tool calls.

The early-submit penalty (`+0.2`) is deliberately positive but low. Previous
iterations used a severely negative penalty (`−2.55`) which made the model
afraid to ever submit, causing mode collapse. The current value is reachable but
clearly worse than gathering evidence first.

## Evidence Alignment

Tool scores are normalized between `0` and `1`. Lower scores mean the tool found
more serious issues. Evidence alignment compares those scores with the verdict:

- `approve` should align with high tool scores
- `request_changes` and `reject` should align with lower tool scores

For example, if `check_security` returns a score near `0.0` because it found a
critical SQL injection, a `reject` verdict receives a strong evidence-alignment
bonus. An `approve` verdict receives an evidence penalty.

## Evidence Penalties

The reward system includes final penalties for contradictions:

- approving with critical findings
- approving with many warnings
- requesting changes when evidence indicates a severe reject-level issue
- rejecting when all gathered tools are clean

This keeps the policy from gaming the final label alone. It must keep the
verdict consistent with the evidence it asked for.

## Example: SQL Injection

Task: `py_sql_injection`

Diff excerpt:

```diff
+def search_users(conn):
+    query = request.args.get("q", "")
+    sql = f"SELECT * FROM users WHERE name LIKE '%{query}%'"
+    return conn.execute(sql).fetchall()
```

Expected verdict: `reject`

Good route:

```json
{"tool_name": "check_security", "arguments": {}}
```

Then:

```json
{
  "tool_name": "submit_review",
  "arguments": {
    "verdict": "reject",
    "confidence": 0.9,
    "reasoning": "Security evidence shows user input is interpolated into SQL."
  }
}
```

Why this scores well:

- the first tool matches the `security` risk domain
- the terminal verdict is correct
- the evidence supports rejection
- the route is short

Bad route:

```json
{"tool_name": "check_tests", "arguments": {}}
```

Then:

```json
{"tool_name": "submit_review", "arguments": {"verdict": "approve"}}
```

Why this scores poorly:

- the tool choice misses the main risk
- approval contradicts the expected verdict
- no security evidence supports the decision

## Why This Reward Shape Works for GRPO

GRPO compares multiple completions for the same prompt. The reward function
creates useful contrast between completions:

- relevant tool call (`+0.3`) beats irrelevant tool call (`+0.1`)
- evidence-backed correct submit (`+1.0`) beats early correct (`+0.2`)
- early correct (`+0.2`) beats wrong verdict (`−0.8`)
- duplicate calls (`−0.4`) are clearly worse than any novel tool

Raw rewards go directly to GRPO without normalization. Previous iterations
normalized rewards into `[0.01, 0.99]`, which compressed the signal so
severely that GRPO could not distinguish between actions (reward variance
collapsed to zero, causing mode collapse). The current raw scale has a total
spread of ~2.5, giving GRPO clear advantage signal.

That contrast is what teaches routing behavior.
