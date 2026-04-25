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

- new tool call: small positive base reward
- relevant tool for the task risk domain: extra bonus
- duplicate tool call: penalty
- too many steps for the task difficulty: efficiency penalty

Example: a security task such as `py_sql_injection` has `risk_domains` containing
`security`, so `check_security` is relevant. A first call to `check_security`
earns more than a first call to an unrelated tool.

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

The key cases are:

- correct verdict with supportive evidence: high reward
- correct verdict without supportive evidence: partial reward
- escalation on risky tasks: small partial credit
- wrong verdict: penalty

This distinction matters. The model should not learn to guess a verdict from the
prompt alone. If the expected verdict is `reject`, and the model rejects only
after gathering security evidence, that is better than rejecting without any
tool call.

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

- security tool then reject beats unsupported reject
- unsupported reject beats approving a critical vulnerability
- one relevant tool beats three redundant tools
- clean approval beats paranoid rejection on safe refactors

That contrast is what teaches routing behavior.
