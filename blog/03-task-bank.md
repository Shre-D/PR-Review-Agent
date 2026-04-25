---
title: "Building a Multi-Language PR Review Benchmark"
date: 2026-04-24
tags: [benchmark, tasks, evaluation, multi-language]
---

# Building a Multi-Language PR Review Benchmark

The hardest part of training an RL agent isn't the model or the training loop. It's designing the task bank. A bad task bank produces agents that learn to game your reward function rather than learn to review code.

Here's how we designed 65+ benchmark tasks that actually test what we care about.

## Design Principles

**1. Tasks must be unambiguous.** Every task has a single ground-truth verdict: `approve`, `request_changes`, or `reject`. There's no "it depends." The diff is crafted so a careful reviewer would consistently reach the same conclusion.

**2. Tasks must test routing, not just verdicts.** The agent must learn *which tools to call*, not just *what to conclude*. A diff with SQL injection should trigger `check_security`. A Dockerfile without a USER directive should trigger `check_config`. A clean Go refactor should trigger minimal tools.

**3. Tasks must span the full language surface.** We cover Python, TypeScript, JavaScript, Java, Go, Rust, Dockerfile, YAML (including Kubernetes and GitHub Actions), and .gitignore. An agent that only knows Python is useless in a polyglot codebase.

**4. Clean diffs must be represented.** If the task bank only has problematic code, the agent learns to always flag issues. We include diffs that should be approved — clean refactors, routine dependency bumps, sensible changes — so the agent learns restraint.

## The Task Schema

```python
@dataclass
class PRTask:
    task_id: str           # unique slug, e.g. "py_sql_injection"
    diff_str: str          # unified diff text
    pr_description: str    # PR title + body
    primary_language: str  # "python", "go", "rust", ...
    changed_file_types: list[str]
    repo_kind: str         # "library" | "service" | "infra" | "mixed"
    expected_verdict: str  # "approve" | "request_changes" | "reject"
    risk_domains: list[str]  # ["security","quality","tests","config","build"]
    difficulty: str        # "easy" | "medium" | "hard"
    review_goal: str       # one-sentence description of what to catch
```

The `risk_domains` field is critical: it drives the grader's `relevant_tools()` function, which determines which tool calls earn bonus reward.

## Representative Tasks

### Security — Reject

**`py_sql_injection`** (easy): A Flask search endpoint that builds SQL queries with f-strings and user input. The fix is obvious to any security-aware reviewer. `check_security` should catch it via heuristics before semgrep even runs.

```diff
+def search_users(conn):
+    query = request.args.get("q", "")
+    sql = f"SELECT * FROM users WHERE name LIKE '%{query}%'"
+    return conn.execute(sql).fetchall()
```

**`ts_eval_template`** (easy): A TypeScript helper that calls `eval()` on user input. Classic XSS/code injection risk.

### Quality — Request Changes

**`java_null_contract`** (medium): Three chained method calls with no null guards. Will NPE in production on any user without a complete profile.

```diff
+public String fullName(User user) {
+    return user.getProfile().getDisplayName().trim();
+}
```

**`rust_unwrap_io`** (easy): `unwrap()` on a file read in a library function. Fine in a test, catastrophic in a server.

### Config — Request Changes

**`docker_root_user`** (medium): A Dockerfile that doesn't add a `USER` directive. The container runs as root. `check_config` should flag it via fixture-backed inspection.

**`gha_permissions_write`** (medium): A GitHub Actions workflow with `contents: write` and `pull-requests: write` on a PR-triggered workflow. Over-broad permissions enable supply chain attacks.

**`gitignore_secret_file`** (easy): A .gitignore change that *removes* the broad `.env*` pattern and replaces it with only `.env.local`. After this change, `.env.production` and `.env.staging` are no longer ignored and could be accidentally committed.

### Clean — Approve

**`go_clean_refactor`** (easy): Extracts a simple validation helper into its own function. No new logic, no new risk. The agent should approve quickly without over-calling tools.

**`java_dependency_bump`** (easy): A Maven pom.xml that bumps slf4j-api from one patch version to another. Routine maintenance. Approve.

## The Difficulty Spectrum

We deliberately include easy, medium, and hard tasks. Easy tasks test whether the agent can identify obvious patterns quickly. Medium tasks require calling multiple tools to gather sufficient evidence. Hard tasks involve subtle issues — the kind that slip through even careful human review.

The reward function reinforces this: the efficiency penalty kicks in after step 4, meaning the agent is rewarded for being decisive on easy tasks and thorough only when warranted.

## Loading Tasks

Tasks live in `tasks/tasks.jsonl` (65+ entries) with a 12-task hardcoded fallback in `server/tasks.py`. The JSONL format enables easy extension — add new tasks by appending a line to the file, no code changes needed.

---

*Next: [Reward Shaping for Code Review](./04-reward-shaping.md)*
