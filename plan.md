# Implementation Plan: Context-Aware PR Review Agent

**Status:** Active  
**Date:** 2026-04-24  
**Goal:** Extend the current benchmark router into a production-grade, org-configurable PR review system with rich verdicts, author-aware review depth, and a dynamic reward system wired to a canonical tool registry.

**Training and routing constraint:** SLM-only in the training and adaptive routing paths. No larger-model calls during rollout, gradient updates, or route selection. Context loading is an offline structural step in the checked-in implementation. See TRAINING.md for GPU setup.

---

## 0. Hackathon Review Findings To Fix Next

These findings came from the repository review on 2026-04-25. Treat them as the immediate pre-work before the larger context-aware architecture changes below.

### 0.1 Submission blockers

- [x] Fix model action parsing in `train/grpo_train.py`.
  - `parse_action()` now scans for valid nested JSON tool-call objects instead of using `rfind("{")`.
  - Tests cover nested tool-call JSON, malformed model output, and non-dict arguments.

- [x] Align `submit_review` arguments across prompts, parser, tools, and benchmarks.
  - `train/grpo_train.py` and `benchmarks/loader_benchmark.py` prompt the model to emit `confidence`.
  - `envs/pr_review_env/server/tools.py::submit_review()` now accepts optional `confidence` and clamps it into `[0.0, 1.0]`.
  - `benchmarks/loader_benchmark.py` reuses the shared parser.

- [x] Make the GRPO path a real end-to-end training/evaluation story.
  - `train/grpo_train.py` now builds replayable environment states and passes an online `reward_funcs` callback to `GRPOTrainer`.
  - Each completion is parsed as a tool call, replayed into `PRReviewEnv`, and scored from the live environment reward.
  - The printed `benchmarks/evaluate_trained_model.py` command was removed because that file is not present.

### 0.2 Benchmark quality fixes

- [x] Recalibrate severity scoring so obvious security issues fail correctly.
  - In heuristic mode, `py_sql_injection` and `ts_eval_template` now receive `check_security` score `0.55`, so `benchmarks/run_baselines.py::decide_final_verdict()` rejects them.
  - Security-critical findings should drive `reject` or at least `request_changes` without requiring multiple findings.

- [x] Improve baseline reproducibility.
  - `benchmarks/evaluate_baselines.py` no longer uses Python `hash()` in seeded random policy selection.
  - It now uses a stable `hashlib.sha256(task_id.encode()).hexdigest()` offset.

- [x] Check in or regenerate claimed benchmark artifacts.
  - `rewards/baseline_eval.json` has been regenerated from the current heuristic backend.

### 0.3 Documentation and config-loader consistency

- [x] Implement or remove the documented context loader path.
  - `python -m envs.pr_review_env.server.context_loader` now generates a structural `review_config.json`.
  - `train/grpo_train.py` now accepts `--review-config` and injects the loaded config into rollout prompts.

- [x] Fix `benchmarks/loader_benchmark.py` task selection.
  - `run_episode_with_config()` now resets directly with `task_id=task.task_id`.

- [x] Reconcile documented external tools with implemented tools.
  - `docs/review-tool.md` lists `bandit`, `gitleaks`, `hadolint`, `actionlint`, `yamllint`, and `mypy`.
  - Current runtime uses heuristics plus optional `semgrep`, `ruff`, `pylint`, `radon`, language build tools, and `pyyaml`.
  - `docs/review-tool.md` now separates implemented external tools from planned external tools.

### 0.4 Verified current state

- `pytest -q`: 32 passed.
- `python benchmarks/run_baselines.py --episodes 10`: completed.
- `PR_REVIEW_TOOL_BACKEND=heuristic python benchmarks/evaluate_baselines.py`: heuristic accuracy `0.662`, mean return `0.651`; random accuracy `0.400`, mean return `-0.029`.

---

## 1. Problem Statement

The current system routes tool calls based on a fixed `DOMAIN_TO_TOOLS` map and static reward weights in `grader.py`. This works for benchmark evaluation but fails in production because:

- Every company has different security rules, architecture constraints, and code standards.
- A junior dev's first PR needs deeper scrutiny than a tech lead's routine refactor.
- The verdict is a bare string (`"approve"` / `"reject"`) — real reviewers need a confidence score, categorised findings, and a summary.
- The reward system is hardcoded to 5 analysis tools with fixed penalty/bonus values — adding a new tool (bandit, trivy, hadolint) requires editing grader.py source.
- The system prompt gives the model no awareness of the organisation's codebase, architecture, or critical paths.

The solution is a **configuration-driven pipeline** that translates human-readable org documentation into a machine-executable review policy, and a **dynamic reward system** that adapts its weights and thresholds from that policy.

---

## 2. Architecture Overview

```
Org documentation                       Review policy
┌───────────────────┐                ┌──────────────────────┐
│ company-guidelines│                │   review_config.json │
│ architecture.md   │──► context_ ──►│                      │
│ design-spec.md    │   loader.py    │  tool_weights        │
│ features.md       │                │  domain_priorities   │
│ review-tool.md    │                │  author_depth        │
└───────────────────┘                │  critical_paths      │
                                     │  custom_rules        │
                                     │  verdict_thresholds  │
                                     │  enabled_tools       │
                                     │  architecture_summary│
                                     └──────────┬───────────┘
                                                │
    PR diff + metadata + author_context         │
         │                                      │
         ▼                                      ▼
┌────────────────────────────────────────────────────────┐
│                    PRReviewEnv                         │
│                                                        │
│  ┌─────────────┐   ┌──────────────┐   ┌────────────┐ │
│  │ tool_registry│   │ grader.py    │   │ pr_review_ │ │
│  │ .py          │   │ (config-     │   │ env.py     │ │
│  │              │   │  aware)      │   │            │ │
│  │ canonical    │   │              │   │ wires      │ │
│  │ tool specs   │──►│ dynamic wts  │──►│ everything │ │
│  │ normalizers  │   │ author depth │   │ together   │ │
│  └─────────────┘   │ crit-path    │   └────────────┘ │
│                     │ multipliers  │                   │
│                     └──────────────┘                   │
└────────────────────────────┬───────────────────────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ adaptive_router.py  │
                  │                     │
                  │ SLM-only route tier │
                  │ small → light tools │
                  │ medium → standard   │
                  │ large → deep tools  │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ PRReviewVerdict     │
                  │                     │
                  │ verdict + confidence│
                  │ summary + findings  │
                  │ critical / warnings │
                  │ suggestions         │
                  └─────────────────────┘
```

---

## 3. Canonical Tool Registry

**File:** `envs/pr_review_env/server/tool_registry.py`  
**Depends on:** nothing (leaf module)  
**Purpose:** Single source of truth for every external analyser the system can invoke — name, languages, install command, output normaliser, default weight.

### 3.1 Data Model

```python
@dataclass(frozen=True)
class ToolSpec:
    name: str                         # canonical name (e.g. "semgrep")
    display_name: str                 # human-friendly (e.g. "Semgrep SAST")
    domains: tuple[str, ...]          # risk domains it covers ("security", "quality", …)
    languages: tuple[str, ...]        # ("all",) or ("python", "go", …)
    binary: str                       # CLI binary name for shutil.which()
    install_hint: str                 # "pip install semgrep"
    default_weight: float             # default contribution to aggregate score
    default_args: tuple[str, ...]     # CLI args template
    output_format: str                # "json", "text", "sarif"
```

### 3.2 Tool Inventory

The registry will hold entries for every tool we support or plan to support. Each entry has a normaliser function that converts the tool's raw CLI output into our standard `{score: float, findings: list[str], severity_counts: dict}` format.

**Security tools:**

| Tool | Languages | Output | Install | Notes |
|------|-----------|--------|---------|-------|
| semgrep | all | SARIF / JSON | `pip install semgrep` | Community rulesets; best general SAST |
| bandit | Python | JSON | `pip install bandit` | Python-specific security linter |
| gitleaks | all (secrets) | JSON | `brew install gitleaks` / binary | Detects hardcoded secrets, API keys |
| trivy | Dockerfile, deps | JSON | `brew install trivy` / binary | CVE scan on images and lock files |
| njsscan | JS/TS | SARIF | `pip install njsscan` | Node.js security scanner |
| gosec | Go | JSON | `go install github.com/securego/gosec/v2/cmd/gosec` | Go security linter |
| cargo-audit | Rust | JSON | `cargo install cargo-audit` | Rust dependency vulnerability scan |

**Quality tools:**

| Tool | Languages | Output | Install | Notes |
|------|-----------|--------|---------|-------|
| pylint | Python | JSON | `pip install pylint` | Style + correctness |
| mypy | Python | text | `pip install mypy` | Type checking |
| ruff | Python | JSON | `pip install ruff` | Fast Python linter (replaces flake8+isort+pyupgrade) |
| radon | Python | JSON | `pip install radon` | Cyclomatic complexity |
| eslint | JS/TS | JSON | `npm install eslint` | JS/TS linter |
| golangci-lint | Go | JSON | binary | Meta-linter for Go |
| cargo clippy | Rust | text | `rustup component add clippy` | Rust lint |
| pmd | Java | XML/JSON | binary | Java static analysis |
| checkstyle | Java | XML | binary | Java style checker |

**Build/type tools:**

| Tool | Languages | Output | Install | Notes |
|------|-----------|--------|---------|-------|
| tsc | TypeScript | text | `npm install typescript` | `tsc --noEmit` |
| javac | Java | text | JDK | Compile check |
| go build / go vet | Go | text | Go SDK | Build + vet |
| cargo check | Rust | text | Rust toolchain | Type + borrow check |
| mypy (strict) | Python | text | `pip install mypy` | Overlaps with quality |

**Config tools:**

| Tool | Languages | Output | Install | Notes |
|------|-----------|--------|---------|-------|
| hadolint | Dockerfile | JSON | binary | Dockerfile linter |
| actionlint | GitHub Actions | text/JSON | binary | GHA workflow linter |
| yamllint | YAML | text | `pip install yamllint` | YAML syntax + style |

### 3.3 Normaliser Contract

Every tool gets a normaliser function with this signature:

```python
def normalise_<tool>(raw_output: str, exit_code: int) -> ToolResult:
    """Convert raw CLI output to standard ToolResult."""
    ...

@dataclass
class ToolResult:
    score: float                      # 0.0 (many issues) to 1.0 (clean)
    findings: list[str]               # human-readable finding strings
    severity_counts: dict[str, int]   # {"critical": 0, "warning": 2, "info": 1}
    raw: str                          # original output for debugging
    tool_name: str                    # which tool produced this
    analysis_mode: str                # "real", "heuristic", "unavailable"
```

### 3.4 Integration with Existing tools.py

The existing `check_security`, `check_quality`, etc. MCP tools remain the agent's action space. Internally, each tool dispatches to one or more `ToolSpec` entries based on:
1. Language of the diff
2. What's installed (via `shutil.which`)
3. What's enabled in `ReviewConfig.enabled_tools`

Example: `check_security` on a Python diff might run `semgrep` + `bandit` + `gitleaks`, aggregate their normalised results, and return a single `{score, findings}` dict.

### 3.5 Dynamic `DOMAIN_TO_TOOLS` Generation

Replace the hardcoded map in `grader.py`:

```python
def build_domain_to_tools(registry: dict[str, ToolSpec]) -> dict[str, set[str]]:
    """Build DOMAIN_TO_TOOLS dynamically from the registry."""
    mapping: dict[str, set[str]] = {}
    for spec in registry.values():
        for domain in spec.domains:
            mapping.setdefault(domain, set()).add(f"check_{domain}")
    return mapping
```

This means adding a new tool to the registry automatically updates the reward system's relevance mapping — no manual grader.py edits.

---

## 4. Context Ingestion Pipeline

**File:** `envs/pr_review_env/server/context_loader.py`  
**Depends on:** `tool_registry.py`  
**Purpose:** Parse org documentation (`.md` files) into a `ReviewConfig` JSON that parameterises the reward system, tool selection, and model prompt.

> **Architectural boundary:** `context_loader.py` is an **offline, one-time step**. The checked-in path is deterministic structural parsing. Any future model-assisted extraction must stay outside per-PR routing and training. The RL router (the trained Qwen3-1.7B) never calls the loader — it only reads the pre-generated `review_config.json`.
>
> ```
> [OFFLINE — runs once]                [ONLINE — runs per PR]
>
> docs/*.md
>     │
>     ▼
> context_loader.py                    review_config.json
>     │  structurally extracts              │
>     │  machine config                     ▼
>     ▼                              PRReviewEnv (grader, tools)
> review_config.json                         │
>                                            ▼
>                                   Qwen3-1.7B SLM router
>                                   (always small, always fast)
> ```

### 4.1 Input: Documentation Directory

```
docs/                              # or any user-specified path
├── company-guidelines.md          # security rules, coding standards, banned patterns
├── architecture.md                # frontend/backend stack, service boundaries
├── design-spec.md                 # naming conventions, API patterns, state management
├── features.md                    # critical features, protected paths, SLAs
└── review-tool.md                 # (optional) explicit tool weights and thresholds
```

A user can provide any subset of these. The loader processes whatever it finds.

### 4.2 Output: ReviewConfig

```python
class CustomRule(BaseModel):
    domain: str                       # "security", "quality", "config", …
    pattern: str                      # regex or keyword to match in diff
    severity: str                     # "critical", "warning", "info"
    message: str                      # human-readable finding message
    languages: list[str]              # [] means all languages

class ReviewConfig(BaseModel):
    # --- Tool control ---
    tool_weights: dict[str, float]    # override TOOL_WEIGHTS from grader.py
    enabled_tools: list[str]          # subset of TOOL_REGISTRY keys; empty = all available

    # --- Verdict thresholds ---
    reject_threshold: float = 0.30    # aggregate score below this → reject
    request_changes_threshold: float = 0.65

    # --- Domain priorities ---
    domain_priorities: dict[str, float] = {}  # multiplier on relevance bonus
                                              # {"security": 1.5} → security bonus is 0.08*1.5=0.12

    # --- Author-level review depth ---
    author_depth: dict[str, float] = {
        "junior": 1.4,                # wider efficiency window (step penalty starts later)
        "mid": 1.0,
        "senior": 0.7,               # tighter window (trust senior code)
        "lead": 0.6,
        "non_tech": 1.2,             # slightly wider (config/doc changes need careful review)
    }

    # --- Protected paths ---
    critical_paths: list[str] = []    # glob patterns: ["auth/**", "payments/**", "migrations/**"]

    # --- Custom rules from guidelines ---
    custom_rules: list[CustomRule] = []

    # --- Architecture context (injected into model prompt) ---
    architecture_summary: str = ""    # 2-3 sentences describing the stack
    # (adaptive router model IDs are in Section 8.4, not here)
```

### 4.3 Two-Pass Extraction

**Pass 1 — Structural parsing** (no LLM required):

If `review-tool.md` exists, parse it for explicit YAML/TOML frontmatter or markdown tables. This is the "override" file — explicit values here take precedence over any inferred values.

```markdown
# review-tool.md

## Tool Weights
| Tool | Weight |
|------|--------|
| check_security | 0.40 |
| check_quality | 0.25 |
| check_tests | 0.20 |
| check_config | 0.15 |

## Thresholds
- reject_threshold: 0.30
- request_changes_threshold: 0.60

## Critical Paths
- auth/
- payments/
- db/migrations/
```

**Pass 2 — Semantic extraction (future, offline only):**

For `company-guidelines.md`, `architecture.md`, `design-spec.md`, `features.md`, a future loader may use the same SLM family to extract structured meaning from natural-language prose. This is not part of adaptive routing and never runs per PR. The current implemented path is structural-only.

Extraction tasks:
1. **Custom rules** from guidelines: "No raw SQL queries in any handler" → `CustomRule(domain="security", pattern="execute\\(.*f['\"]", severity="critical", message="Raw SQL detected — use parameterised queries")`
2. **Architecture summary** from architecture docs: "React+TypeScript frontend, FastAPI+PostgreSQL backend, Redis cache layer" → stored as `architecture_summary` for prompt injection.
3. **Domain priorities** from features docs: if the doc emphasises "security is our top priority" → `domain_priorities={"security": 1.5}`.

The extraction prompt (same regardless of model size):

```
Given the following organisation documentation, extract:
1. Custom code review rules (pattern + severity + message)
2. A 2-sentence architecture summary
3. Domain priority rankings (security/quality/tests/config/build)
4. Critical file paths that must always receive deep review

Output valid JSON matching this schema: { ... }

Document:
<contents>
```

### 4.4 CLI Interface

```bash
# Structural only — no model, fast, deterministic
python -m envs.pr_review_env.server.context_loader \
  --docs-dir ./docs/ \
  --output review_config.json

# Structural extraction with a compatibility flag for future rule extraction
python -m envs.pr_review_env.server.context_loader \
  --docs-dir ./docs/ \
  --output review_config.json \
  --extract-rules
```

This is a developer setup command, not per-PR automation. Run it when docs change.

### 4.5 Caching

The config is written to `review_config.json` and loaded **once on server startup**. It does not change during a training run or evaluation. If the org docs change, a developer re-runs the loader and restarts the server. The RL router reads a static JSON file and does not perform context extraction.

---

## 5. Author Context

**Files modified:** `models.py`, `tasks.py`, `pr_review_env.py`  
**Purpose:** Make the review depth sensitive to who submitted the PR.

### 5.1 Data Model

```python
class AuthorContext(BaseModel):
    level: Literal["junior", "mid", "senior", "lead", "non_tech", "unknown"] = "unknown"
    github_username: str = ""
    account_age_days: int = 0         # days since first commit in this repo
    total_commits: int = 0            # total commits to this repo
    recent_revert_rate: float = 0.0   # fraction of their recent PRs that were reverted
    is_first_pr: bool = False
    team: str = ""                    # "frontend", "backend", "infra", "data", ""
```

### 5.2 How Author Level Is Determined

In production, author level comes from GitHub API data or org config. For the benchmark, each `PRTask` gets a new field:

```python
@dataclass
class PRTask:
    # ... existing fields ...
    author_level: str = "mid"         # default for benchmark tasks
```

The training loop and baselines pass this through to the grader. The grader uses `ReviewConfig.author_depth[author_level]` to adjust the efficiency window:

```python
depth = config.author_depth.get(task.author_level, 1.0) if config else 1.0
efficiency_start = int(4 * depth)   # junior: step 5-6, senior: step 3
```

### 5.3 Observation Extension

Add to `PRReviewObservation`:

```python
class PRReviewObservation(Observation):
    # ... existing fields ...
    author_context: AuthorContext = Field(default_factory=AuthorContext)
    critical_paths_touched: list[str] = Field(default_factory=list)
    estimated_risk_level: str = "medium"  # computed from config + diff + author
```

The model's system prompt gains awareness:

```
Author: junior developer (first PR in payments/ directory)
Risk: HIGH — critical path touched + junior author
```

---

## 6. Rich Verdict Output

**Files modified:** `models.py`, `tools.py` (`submit_review` tool), `pr_review_env.py`  
**Purpose:** Replace the bare `verdict: str` with a structured review document.

### 6.1 Data Model

```python
class Finding(BaseModel):
    domain: str                       # "security", "quality", …
    tool: str                         # which tool found it
    message: str                      # human-readable description
    file: str = ""                    # file path if known
    line: int = 0                     # line number if known
    severity: str = "warning"         # "critical", "warning", "info"
    rule_id: str = ""                 # e.g. "semgrep:python.flask.security.sqli"

class PRReviewVerdict(BaseModel):
    verdict: Literal["approve", "request_changes", "reject", "escalate"]
    confidence: float                 # 0.0 – 1.0
    summary: str                      # 2-3 sentence human-readable summary
    critical_findings: list[Finding]  # must-fix before merge
    warnings: list[Finding]           # should-fix, not blocking
    suggestions: list[Finding]        # nice-to-have improvements
    tools_used: list[str]             # tools called during this review
    aggregate_score: float            # weighted score from grader
    review_context: str               # "Reviewed as junior-authored PR; extra scrutiny on auth/ paths"
```

### 6.2 How the Model Produces Rich Verdicts

The `submit_review` action schema expands. The model now outputs:

```json
{
  "tool_name": "submit_review",
  "arguments": {
    "verdict": "reject",
    "confidence": 0.92,
    "reasoning": "SQL injection via unsanitised user input in search endpoint"
  }
}
```

The environment constructs the full `PRReviewVerdict` by combining:
- Model's verdict + confidence + reasoning → `verdict`, `confidence`, `summary`
- Accumulated tool results → `critical_findings`, `warnings`, `suggestions` (classified by severity from tool results)
- State metadata → `tools_used`, `aggregate_score`, `review_context`

### 6.3 Confidence-Aware Reward

Terminal reward uses confidence as a signal:

```python
def terminal_reward(task, verdict: PRReviewVerdict, config: ReviewConfig) -> float:
    correct = verdict.verdict == task.expected_verdict

    if correct:
        # High confidence + correct = full reward
        # Low confidence + correct = reduced (model should learn to be confident when right)
        confidence_mult = 0.7 + 0.3 * verdict.confidence  # range [0.7, 1.0]
        evidence = verdict.aggregate_score
        supportive = len(set(verdict.tools_used) & relevant_tools(task))
        if supportive >= 1:
            return round(confidence_mult * (1.0 + min(0.25, evidence * 0.25)), 3)
        return round(confidence_mult * 0.35, 3)

    # Wrong verdict — overconfident wrong answers are punished harder
    overconfidence_penalty = -0.55 * (1.0 + 0.3 * verdict.confidence)  # [-0.55, -0.715]
    return round(overconfidence_penalty, 3)
```

This teaches the model:
- Be confident when right → higher reward
- Be uncertain when wrong → less punishment than being confident and wrong
- Never bluff: high confidence + wrong verdict is the worst outcome

---

## 7. Dynamic Reward System

**File modified:** `envs/pr_review_env/server/grader.py`  
**Purpose:** Make every reward parameter configurable through `ReviewConfig`.

### 7.1 Config-Driven `step_reward`

```python
def step_reward(
    task: PRTask,
    tool_name: str,
    already_called: bool,
    step_count: int,
    config: ReviewConfig | None = None,
) -> float:
    if tool_name in {"submit_review", "escalate"}:
        return 0.0

    if already_called:
        return -0.20

    base = 0.05
    is_relevant = tool_name in relevant_tools(task, config)

    # Domain priority multiplier
    bonus = 0.08
    if config and is_relevant:
        domain = _tool_to_primary_domain(tool_name)
        bonus *= config.domain_priorities.get(domain, 1.0)

    reward = base + (bonus if is_relevant else 0.0)

    # Author-depth-adjusted efficiency window
    depth = 1.0
    if config and hasattr(task, "author_level"):
        depth = config.author_depth.get(task.author_level, 1.0)
    eff_start = int(4 * depth)
    if step_count > eff_start:
        reward -= 0.03 * (step_count - eff_start)

    return round(reward, 3)
```

### 7.2 Config-Driven `terminal_reward`

```python
def terminal_reward(
    task: PRTask,
    submitted_verdict: str,
    tool_results: dict[str, dict],
    config: ReviewConfig | None = None,
) -> float:
    # ... existing logic ...

    # Critical path multiplier
    is_critical = _touches_critical_path(task, config)
    correct_mult = 1.3 if is_critical else 1.0
    wrong_mult = 1.5 if is_critical else 1.0

    if correct:
        if supportive >= 1:
            return round(correct_mult * (1.0 + min(0.25, evidence * 0.25)), 3)
        return round(correct_mult * 0.35, 3)

    return round(-0.55 * wrong_mult, 3)
```

### 7.3 Config-Driven `aggregate_tool_scores`

Tool weights come from config instead of the hardcoded `TOOL_WEIGHTS` dict:

```python
def aggregate_tool_scores(
    tool_results: dict[str, dict],
    config: ReviewConfig | None = None,
) -> float:
    weights = TOOL_WEIGHTS  # default fallback
    if config and config.tool_weights:
        weights = config.tool_weights

    # ... same weighted average logic, using dynamic weights ...
```

### 7.4 Custom Rules as Additional Findings

Custom rules from `ReviewConfig.custom_rules` are evaluated during tool execution. When `check_security` runs, it also checks all custom rules with `domain="security"` against the diff:

```python
def _apply_custom_rules(diff_str: str, domain: str, config: ReviewConfig) -> list[str]:
    findings = []
    for rule in config.custom_rules:
        if rule.domain != domain:
            continue
        if rule.languages and task_language not in rule.languages:
            continue
        if re.search(rule.pattern, diff_str, re.MULTILINE | re.IGNORECASE):
            findings.append(f"[{rule.severity.upper()}] {rule.message}")
    return findings
```

---

## 8. Adaptive SLM Router

**File:** `train/adaptive_router.py`  
**Depends on:** `models.py`, `context_loader.py`  
**Purpose:** Use the SLM to adapt review depth, required evidence, and tool budget without switching to larger models.

> **Core rule:** The RL router is **always a SLM** (the GRPO-trained Qwen3-1.7B). Training and inference use this model only. The adaptive layer never calls 7B/API models; it only chooses route depth and evidence requirements.

### 8.1 Route Tiers Control Process, Not Model Size

```
                    PR arrives
                        │
                        ▼
              ┌─────────────────────┐
              │  select_tier()      │  ← purely a complexity signal,
              │  (heuristic, fast)  │    not a model call
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │  ALWAYS runs        │
              │  Qwen3-1.7B SLM     │  ← the only trained RL router
              │  (GRPO checkpoint)  │
              └──────────┬──────────┘
                         │ SLM verdict + confidence
                         │
              ┌──────────▼──────────┐
              │  tier + confidence  │
              │  decide evidence    │
              │  requirements       │
              └──────────┬──────────┘
                    ┌────┴────┐
              harder path   light path
                    │          │
                    ▼          ▼
             more tools     fewer tools
             tighter gate   faster verdict
```

The SLM always runs. For harder or uncertain PRs, the router increases tool budget and evidence requirements instead of switching model class. This means:
- Training data and reward shaping are unchanged — always SLM
- For routine PRs, the SLM can use a short evidence path
- For large/critical PRs, the SLM must gather more evidence before final verdict

### 8.2 Tier Selection (Complexity Signal Only)

`select_tier()` is a cheap heuristic — no model call, no LLM, just metadata:

```python
class ModelTier(Enum):
    SMALL = "small"     # light evidence path
    MEDIUM = "medium"   # standard evidence path
    LARGE = "large"     # deep evidence path, still SLM-only

def select_tier(obs: PRReviewObservation, config: ReviewConfig) -> ModelTier:
    loc_added = obs.diff_str.count("\n+")
    num_files = len(obs.changed_file_types)
    is_critical = bool(obs.critical_paths_touched)
    is_junior = obs.author_context.level in ("junior", "non_tech")

    # Critical path + junior author → require highest evidence depth
    if is_critical and is_junior:
        return ModelTier.LARGE

    # Large or multi-language diffs → require deeper evidence
    if loc_added > 800 or (len(set(obs.changed_file_types)) > 2 and loc_added > 300):
        return ModelTier.LARGE

    if loc_added > 200 or num_files > 1 or is_critical:
        return ModelTier.MEDIUM

    return ModelTier.SMALL
```

### 8.3 Route Requirements

```python
def route_requirements(tier: ModelTier) -> dict[str, int | bool]:
    """Return SLM-only route requirements."""
    if tier == ModelTier.SMALL:
        return {"min_tools": 1, "max_steps": 3, "requires_security": False}
    if tier == ModelTier.MEDIUM:
        return {"min_tools": 2, "max_steps": 5, "requires_security": True}
    return {"min_tools": 3, "max_steps": 7, "requires_security": True}
```

### 8.4 Config Fields for Adaptive Routing

There is only one policy model in `ReviewConfig`:

```python
class ReviewConfig(BaseModel):
    policy_model: str = "Qwen/Qwen3-1.7B"
    escalation_confidence_threshold: float = 0.6
```

Training and inference both use this SLM. Route tiers control process depth, not model class.

---

## 9. Updated System Prompt

The model's system prompt becomes config-aware. `build_obs_prompt` injects org context:

```python
def build_system_prompt(config: ReviewConfig | None = None) -> str:
    base = """\
You are a code reviewer. For each step, output a JSON tool call:
{"tool_name": "<name>", "arguments": {<args>}}

Available tools: check_security, check_quality, check_build_and_types,
check_tests, check_config, submit_review, escalate.

For submit_review use:
{"tool_name": "submit_review", "arguments": {"verdict": "<verdict>", "confidence": <0-1>, "reasoning": "<brief>"}}
"""
    if config and config.architecture_summary:
        base += f"\nRepository architecture: {config.architecture_summary}\n"

    if config and config.critical_paths:
        base += f"\nCritical paths (require extra scrutiny): {', '.join(config.critical_paths)}\n"

    return base

def build_obs_prompt(obs: PRReviewObservation) -> str:
    parts = [
        f"Language: {obs.primary_language}",
        f"Files: {', '.join(obs.changed_file_types)}",
    ]
    if obs.author_context and obs.author_context.level != "unknown":
        parts.append(f"Author: {obs.author_context.level} developer")
    if obs.critical_paths_touched:
        parts.append(f"CRITICAL PATHS TOUCHED: {', '.join(obs.critical_paths_touched)}")
    if obs.estimated_risk_level in ("high", "critical"):
        parts.append(f"Risk level: {obs.estimated_risk_level.upper()}")

    parts += [
        "",
        f"PR: {obs.pr_description}",
        "",
        "Diff:",
        obs.diff_str,
    ]
    if obs.review_history:
        parts += ["", "Review so far:"] + obs.review_history
    return "\n".join(parts)
```

---

## 10. Data Flow: End-to-End Example

**Setup:**
1. User runs `context_loader.py --docs-dir ./docs/ --output review_config.json`
2. This produces a config with `tool_weights`, `critical_paths=["auth/", "payments/"]`, `domain_priorities={"security": 1.5}`, `author_depth`, and 3 custom rules.

**Episode:**
1. **`reset()`** — env loads a task. Task has `author_level="junior"`, diff touches `auth/login.py`. Config says `auth/` is critical.
2. Observation includes `author_context.level="junior"`, `critical_paths_touched=["auth/login.py"]`, `estimated_risk_level="high"`.
3. **Adaptive router** sees critical + junior → selects `ModelTier.LARGE`.
4. Model sees system prompt with architecture context + "CRITICAL PATHS TOUCHED: auth/login.py" + "Author: junior developer".
5. **Step 1:** Model calls `check_security`. Step reward: `0.05 + 0.08 * 1.5 = +0.17` (security priority 1.5x). Custom rules also run; one fires on a regex match → extra finding.
6. **Step 2:** Model calls `check_quality`. Step reward: `+0.05` (quality not in risk_domains for this task).
7. **Step 3:** Model calls `submit_review` with `{"verdict": "reject", "confidence": 0.91, "reasoning": "SQL injection in auth handler"}`.
8. Environment builds `PRReviewVerdict`: verdict=reject, confidence=0.91, 2 critical findings from check_security (including the custom rule match), 1 warning from check_quality, aggregate_score=0.82.
9. **Terminal reward:** correct + critical path mult 1.3 × (1.0 + 0.25 * 0.82 * 0.25) = 1.3 × 1.05 = **+1.37**. Plus confidence multiplier.
10. Full episode return: 0.17 + 0.05 + 1.37 = **+1.59**.

Compare to the same episode without config: 0.13 + 0.05 + 1.0 = +1.18. The config adds +0.41 reward by correctly prioritising security on a critical path.

---

## 11. File Manifest

| File | Action | Description |
|------|--------|-------------|
| `envs/pr_review_env/server/tool_registry.py` | **Done** | ToolSpec dataclass, ToolResult dataclass, TOOL_REGISTRY dict, lookup helpers, normaliser |
| `envs/pr_review_env/server/context_loader.py` | **Done** | ReviewConfig validation, structural docs parser, CLI entrypoint |
| `envs/pr_review_env/models.py` | **Done** | Added AuthorContext, Finding, PRReviewVerdict, ReviewConfig; extended PRReviewObservation with author_context, critical_paths_touched, estimated_risk_level |
| `envs/pr_review_env/server/grader.py` | **Done** | Accepts ReviewConfig in step_reward, terminal_reward, aggregate_tool_scores; adds confidence-aware reward, author-depth efficiency, dynamic DOMAIN_TO_TOOLS |
| `envs/pr_review_env/server/tools.py` | **Done** | Custom rules evaluate inside existing check_* functions; submit_review accepts confidence; heuristic behavior preserved |
| `envs/pr_review_env/server/tasks.py` | **Done** | Added author_level field to PRTask; fallback tasks remain compatible |
| `envs/pr_review_env/server/pr_review_env.py` | **Done** | Loads ReviewConfig on init; passes to grader; computes critical_paths_touched; populates author_context and final verdict |
| `train/adaptive_router.py` | **Done** | ModelTier enum, select_tier(), route_requirements(), prompt_route_context(); SLM-only |
| `train/grpo_train.py` | **Done** | Loads ReviewConfig; uses config-aware system prompt; includes SLM-only route requirements |
| `docs/review-tool.md` | **Create** | Example org config file with tool weights, thresholds, critical paths |
| `docs/company-guidelines.md` | **Create** | Example org guidelines for testing the context loader |
| `tasks/tasks.jsonl` | **Modify** | Add author_level field to existing tasks |

---

## 12. Build Order

Each step should be verified before proceeding to the next.

| Step | Target | Dependencies | Verification |
|------|--------|-------------|--------------|
| 1 | `tool_registry.py` | None | **Done** — import clean; registry tests pass |
| 2 | `context_loader.py` | tool_registry | **Done** — `python -m envs.pr_review_env.server.context_loader --docs-dir docs/ --output /tmp/test_config.json` produces valid JSON |
| 3 | Update `models.py` | None | **Done** — AuthorContext, Finding, PRReviewVerdict, ReviewConfig instantiate |
| 4 | Update `tasks.py` | models.py | **Done** — `author_level` field present; existing tests pass |
| 5 | Update `grader.py` | models, context_loader | **Done** — config-aware path covered by tests; config=None baseline still runs |
| 6 | Update `tools.py` | tool_registry, context_loader | **Done** — custom rules fire on matching diffs; existing heuristics unchanged |
| 7 | Update `pr_review_env.py` | all above | **Done** — full episode cycle with config; `PRReviewVerdict` in terminal observation |
| 8 | `adaptive_router.py` | models | **Done** — SLM-only tier selection logic covered by tests |
| 9 | Update `grpo_train.py` | adaptive_router, context_loader | **Done** — config-aware prompt includes SLM-only route requirements |
| 10 | Example docs + updated tasks | context_loader | **Done** — docs → config → episode → rich verdict covered by tests |

---

## 13. What This Does NOT Change

- **Benchmark evaluation protocol**: Baselines still run with `config=None`, producing identical numbers.
- **OpenEnv API contract**: HTTP endpoints unchanged (`/reset`, `/step`, `/state`, `/health`).
- **MCP tool names**: Still 7 tools. The agent's action space is unchanged.
- **Training algorithm**: GRPO with the same hyperparameters. Config affects rewards, not the optimiser.
- **Backward compatibility**: Every new parameter has a default that reproduces current behaviour.

---

## 14. Training Optimisation Notes

### What is built now (vs plan)

| Deliverable | Status |
|-------------|--------|
| `train/train_config.py` — hardware presets (T4/A100/V100/CPU), QLoRA defaults | **Done** |
| `TRAINING.md` — full GPU setup guide for Colab/Kaggle/RunPod/Lambda | **Done** |
| `docs/` — company-guidelines, architecture, design-spec, features, review-tool.md | **Done** |
| `tasks/comprehensive_tasks.py` — 13 multi-file comprehensive tasks | **Done** |
| `tasks/comprehensive_tasks.jsonl` — serialised JSONL | **Done** |
| `benchmarks/loader_benchmark.py` — structural loader benchmark and checkpoint comparison | **Done** |
| `envs/pr_review_env/server/tasks.py` — `author_level` field added to PRTask | **Done** |
| `train/grpo_train.py` — QLoRA + parallel rollouts | **Pending** |

### Training speed targets

| Platform | Tasks | Generations | Est. epoch | Total (2 epochs) |
|----------|-------|-------------|------------|------------------|
| Colab T4 | 40 | 4 | ~45 min | ~90 min |
| Kaggle T4 | 40 | 4 | ~45 min | ~90 min |
| A100 | 65 | 8 | ~20 min | ~40 min |

### Context features during training

The following features are deterministic and SLM-compatible. They activate when a
generated `review_config.json` is passed through `--review-config` or loaded by
the environment:

- Architecture summary injection into system prompt  
- Custom rule evaluation during tool execution
- Critical path multipliers (reward shaping)
- Author-depth efficiency window adjustment

During training, the environment uses `PR_REVIEW_TOOL_BACKEND=heuristic` and
default reward weights from `grader.py`. This keeps each rollout to ~2-4s on GPU.

---

## 15. Open Questions

1. **Custom rule evaluation cost**: Regex-matching N custom rules on every tool call adds O(N×diff_length) overhead. For N < 100 this is negligible. For large rule sets, pre-compile the patterns at config load time.

2. **Confidence calibration**: The model's self-reported confidence may not correlate with actual accuracy initially. May need Platt scaling or temperature scaling after training.

3. **Config versioning**: When review_config.json changes between training runs, version-tag and store alongside the checkpoint so results remain reproducible.

4. **Loader benchmark timing**: Running checkpoint-based comparison across the full episode suite can take 1-3 hours depending on hardware. Schedule as a separate job after training, not part of the main eval loop.

5. **Author level in benchmark tasks**: The 65 existing tasks default to `author_level="mid"`. Assign realistic levels in a follow-up pass once training is validated — don't block training on this.
