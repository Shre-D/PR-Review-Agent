---
title: "Five Tools That Make Our Reviewer Smart"
date: 2026-04-24
tags: [tools, fastmcp, semgrep, heuristics, static-analysis]
---

# Five Tools That Make Our Reviewer Smart

The agent's analytical power comes from five tools. Each tool returns a score between 0 and 1, a list of findings, and metadata about which backend ran. The agent sees all of this in its observation and uses it to decide whether to call more tools or submit a verdict.

## The Heuristic-First Design

Every tool runs in one of three modes, controlled by `PR_REVIEW_TOOL_BACKEND`:

- **`heuristic`**: Pattern matching on the diff text only. No external binaries. Always available. Fast (~5ms per call).
- **`hybrid`** (default): Tries real tools (semgrep, pylint, etc.) when available; falls back to heuristic.
- **`real`**: Requires external tools; fails if they're not found.

This design means the environment works in any Python environment — CI, local dev, Docker without network access — while still leveraging real analyzers when available.

## Tool 1: `check_security`

**What it catches:** SQL injection, eval(), pickle deserialization, hardcoded credentials, Java runtime exec, shell injection patterns, over-broad GitHub Actions permissions.

```python
@mcp.tool
def check_security(diff_str: str, task_id: str = "") -> dict:
    heuristics = _heuristic_findings(diff_str)["security"]
    with _analysis_targets(diff_str, task_id=task_id) as targets:
        semgrep_payload, semgrep_findings = _semgrep_scan(targets)
        combined = list(heuristics) + semgrep_findings
        return {
            "score": _score_from_findings(combined, penalty=0.2),
            "findings": combined,
            "analysis_mode": targets["analysis_mode"],
        }
```

The heuristic patterns are intentionally multi-signal: SQL injection requires seeing both `request.args` and an f-string SQL pattern. This reduces false positives on benign code that happens to mention database queries.

**Score formula:** `max(0, 1.0 - 0.2 × num_findings)`. Five findings = score 0.0. This steep penalty reflects that security issues are binary: one critical finding is disqualifying.

## Tool 2: `check_quality`

**What it catches:** Unsafe error handling (unwrap(), chained null access), excessive branching, TODO/FIXME markers in production code.

In hybrid mode, it also runs:
- **ruff**: Fast Python linter for style and correctness
- **pylint**: More thorough Python analysis
- **radon**: Cyclomatic complexity analysis

Quality issues use a gentler penalty (`0.16` per finding vs `0.20` for security), reflecting that quality issues warrant `request_changes` rather than `reject`.

## Tool 3: `check_build_and_types`

**What it catches:** TypeScript compilation errors (via `tsc --noEmit`), Java compilation failures (`javac`), Go test failures (`go test ./...`), Cargo check failures.

This tool is unique: it actually *compiles* the code in the fixture workspace. If the PR introduces a type error, this tool will find it. It also detects dependency changes (`package.json`, `pom.xml`, `go.mod`, `Cargo.toml`).

```python
def _build_and_type_findings(targets, file_types):
    if "typescript" in file_types:
        result = _run_command(["tsc", "--noEmit", "-p", "."], cwd=workspace)
        if result["returncode"] != 0:
            findings.append("TypeScript compilation failed")
    if "go" in file_types:
        result = _run_command(["go", "test", "./..."], cwd=workspace)
        ...
```

When the real compilers aren't available, the tool falls back to heuristics: detecting build manifest changes and noting that language changes "may require build validation."

## Tool 4: `check_tests`

**What it catches:** Behavior changes without visible test coverage. Config-only changes that still need CI confidence.

The test check is heuristic-first: it looks for new function/method definitions without corresponding test files. In fixture-backed mode, it actually scans the workspace for `test_*.py`, `*_test.go`, `*.spec.ts` files and suppresses the coverage warning if tests are present.

```python
if test_files:
    findings = [f for f in findings if "without visible test coverage" not in f]
elif any(item in result["file_types"] for item in REPO_GLOBAL_FILE_TYPES):
    findings.append("Config-only changes still need CI confidence or rollback clarity.")
```

## Tool 5: `check_config`

**What it catches:** YAML parse errors, Dockerfile anti-patterns (`:latest` tag, no USER directive), GitHub Actions over-broad permissions, .gitignore narrowing that exposes sensitive files.

The gitignore heuristic deserves special mention. It doesn't just look at what's being *added* — it inspects the *removed* lines:

```python
removed_lines = [
    line[1:].strip()
    for line in diff_str.splitlines()
    if line.startswith("-") and not line.startswith("---")
]
if ".gitignore" in lower and any(
    ".env" in rl or "secret" in rl for rl in removed_lines
):
    findings["config"].append(
        "Gitignore narrowing removes broad env/secret protection"
    )
```

This catches the subtle case where a developer replaces `.env*` with `.env.local` — leaving `.env.production` unprotected — without the diff adding anything that looks dangerous.

## The Scoring Formula

All tools use the same base formula:

```python
def _score_from_findings(findings, base=1.0, penalty=0.18) -> float:
    return max(0.0, round(base - penalty * len(findings), 3))
```

Each tool tunes the `penalty` to reflect severity:
- `check_security`: penalty=0.20 (steep — security is critical)
- `check_quality`: penalty=0.16
- `check_build_and_types`: penalty=0.14
- `check_tests`: penalty=0.12
- `check_config`: penalty=0.17

The grader then aggregates these weighted scores into an `aggregate_score` that feeds into the terminal reward bonus.

---

*Next: [GRPO: Teaching a 1.7B Model to Route Tool Calls](./06-grpo-training.md)*
