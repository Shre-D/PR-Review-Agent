from __future__ import annotations

from typing import Any

from ..models import ReviewConfig
from .tasks import PRTask
from .tool_registry import build_domain_to_tools
from train.adaptive_router import ModelTier, route_requirements

# Valid risk_domains: {"security", "quality", "tests", "config", "build"}
# Valid tool names:   {"check_security", "check_quality", "check_build_and_types", "check_tests", "check_config"}

TOOL_WEIGHTS = {
    "check_security":        0.28,
    "check_quality":         0.20,
    "check_build_and_types": 0.18,
    "check_tests":           0.16,
    "check_config":          0.18,
}

TIER_DECAY_START = {
    "small": 2,
    "easy": 2,
    "medium": 4,
    "large": 6,
    "hard": 6,
}

RAW_MIN = -1.5
RAW_MAX = 1.5
RAW_FLOOR = RAW_MIN
RAW_NEAR_FLOOR = -1.0

_TERMINAL_TOOLS = {"submit_review", "escalate"}


def _evidence_only(tool_results: dict[str, dict]) -> dict[str, dict]:
    """Strip terminal tool entries before scoring/counting evidence.

    Terminal calls (`submit_review`, `escalate`) are not evidence — they're the
    verdict event. Including them in `tool_results` would let `len(...) >=
    min_tools` pass with a single real evidence tool plus the submit, and
    would dilute `aggregate_tool_scores` because terminal payloads carry no
    `score` field.
    """
    return {tool: result for tool, result in tool_results.items() if tool not in _TERMINAL_TOOLS}

# Maps each valid risk_domain to the tools that cover it
DOMAIN_TO_TOOLS: dict[str, set[str]] = {
    "security": {"check_security"},
    "quality":  {"check_quality"},
    "tests":    {"check_tests"},
    "config":   {"check_config"},
    "build":    {"check_build_and_types", "check_config"},
}


def _tool_weights(config: ReviewConfig | None = None) -> dict[str, float]:
    if config and config.tool_weights:
        return {**TOOL_WEIGHTS, **config.tool_weights}
    return TOOL_WEIGHTS


def _domain_to_tools(config: ReviewConfig | None = None) -> dict[str, set[str]]:
    dynamic = build_domain_to_tools()
    mapping = {**DOMAIN_TO_TOOLS}
    for domain, tools in dynamic.items():
        mapping.setdefault(domain, set()).update(tools)
    if config and config.enabled_tools:
        # ReviewConfig enabled_tools names external analyzers. The MCP action
        # space stays stable, so this currently keeps domain mappings intact.
        return mapping
    return mapping


def _tool_score(result: dict[str, Any] | None) -> float:
    """Extract normalised 0-1 score from a tool result dict."""
    if not result:
        return 0.0
    value = result.get("score", 0.0)
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def tier_decay_start(task: PRTask) -> int:
    """Return the tier-aware efficiency decay threshold from rewards.md."""
    tier = getattr(task, "tier", None) or getattr(task, "difficulty", "medium")
    return TIER_DECAY_START.get(str(tier).strip().lower(), 4)


def reward_floor() -> float:
    return normalize_reward(RAW_FLOOR)


def reward_near_floor() -> float:
    return normalize_reward(RAW_NEAR_FLOOR)


def route_min_tools(task: PRTask, config: ReviewConfig | None = None) -> int:
    tier_text = getattr(task, "tier", None) or getattr(task, "difficulty", "medium")
    normalized = str(tier_text).strip().lower()
    tier = (
        ModelTier.SMALL
        if normalized in {"small", "easy"}
        else ModelTier.LARGE
        if normalized in {"large", "hard"}
        else ModelTier.MEDIUM
    )
    req = route_requirements(tier, None, config)
    return int(req["min_tools"])


def relevant_tools(task: PRTask, config: ReviewConfig | None = None) -> set[str]:
    """Return tool names relevant to this task's risk_domains.
    If risk_domains is empty (clean diff), return {"check_quality"} as minimal baseline.
    Otherwise union DOMAIN_TO_TOOLS[d] for each d in task.risk_domains.
    Do NOT reference "ci" or "clean".
    """
    risk_domains = [domain for domain in task.risk_domains if domain not in {"clean", "ci"}]
    if not risk_domains:
        return {"check_quality"}
    result: set[str] = set()
    domain_mapping = _domain_to_tools(config)
    for domain in risk_domains:
        tools = domain_mapping.get(domain)
        if tools:
            result |= tools
    return result or {"check_quality"}


def aggregate_tool_scores(
    tool_results: dict[str, dict[str, Any]],
    task: PRTask | None = None,
    config: ReviewConfig | None = None,
) -> float:
    """Weighted average of 0-1 tool scores.

    For reward grading, relevant tools are weighted 1.5x as specified in
    rewards.md. The older config/tool-weight path is retained for callers that
    do not provide a task.
    """
    if isinstance(task, ReviewConfig) and config is None:
        config = task
        task = None

    tool_results = _evidence_only(tool_results)
    total = 0.0
    used_weight = 0.0
    relevant = relevant_tools(task, config) if task is not None else set()
    weights = _tool_weights(config)
    for tool_name, result in tool_results.items():
        if task is not None:
            weight = 1.5 if tool_name in relevant else 1.0
        else:
            weight = weights.get(tool_name)
            if weight is None:
                continue
        total += weight * _tool_score(result)
        used_weight += weight

    if used_weight == 0.0:
        return 0.0
    return round(total / used_weight, 3)


def evidence_alignment_bonus(
    task: PRTask,
    tool_results: dict[str, dict[str, Any]],
    verdict: str,
    config: ReviewConfig | None = None,
) -> float:
    score = aggregate_tool_scores(tool_results, task, config)
    normalized = verdict.strip().lower()
    if normalized == "approve":
        alignment = score
    elif normalized in {"reject", "request_changes"}:
        alignment = 1.0 - score
    else:
        alignment = 0.5
    return round(min(0.25, alignment * 0.25), 3)


def step_reward(
    task: PRTask,
    tool_name: str,
    already_called: bool,
    step_count: int,
    config: ReviewConfig | None = None,
    evidence_count: int | None = None,
    min_tools: int | None = None,
) -> float:
    """Per-step raw reward for calling a tool (not submit_review/escalate).

    These values are returned directly to GRPO — no normalization. The gaps
    must be wide enough for GRPO to rank actions after advantage computation.

    Scale (raw):
        submit/escalate  ->  0.0  (scored via terminal_reward instead)
        duplicate        -> -0.4  (clearly worse than any novel tool)
        novel irrelevant ->  0.1  (ok, but noticeably below relevant)
        novel relevant   ->  0.3  (+ domain priority scaling)
        after min_tools  -> -0.3  (discourages tool loops once ready)
        efficiency decay  -> -0.05 per step past tier threshold
    """
    if tool_name in {"submit_review", "escalate"}:
        return 0.0

    if already_called:
        return -0.4

    if (
        evidence_count is not None
        and min_tools is not None
        and evidence_count >= min_tools
    ):
        return -0.3

    reward = 0.1
    if tool_name in relevant_tools(task, config):
        domain_bonus = 0.2
        if config:
            matching_domains = [
                domain
                for domain in task.risk_domains
                if tool_name in _domain_to_tools(config).get(domain, set())
            ]
            if matching_domains:
                priority = max(config.domain_priorities.get(domain, 1.0) for domain in matching_domains)
                domain_bonus *= priority
        reward += domain_bonus

    decay_start = tier_decay_start(task)
    if step_count > decay_start:
        reward -= 0.05 * (step_count - decay_start)

    return round(max(-0.1, reward), 3)


def terminal_reward(
    task: PRTask,
    submitted_verdict: str,
    tool_results: dict[str, dict[str, Any]],
    config: ReviewConfig | None = None,
    confidence: float | None = None,
    min_tools: int | None = None,
) -> float:
    """Raw terminal reward on submit_review or escalate.

    Scale (raw) — designed so GRPO sees clear advantage gaps:
        correct + evidence-backed   ->  +1.0 to +1.4
        correct + early submit      ->  +0.2  (reachable, but much worse)
        correct + no supportive     ->  -0.3
        escalate (high-risk wrong)  ->  -0.2 / -0.4
        wrong + evidence            ->  -0.8
        wrong + early               ->  -1.0
    """
    normalized = submitted_verdict.strip().lower()
    expected = task.expected_verdict.strip().lower()
    evidence_results = _evidence_only(tool_results)
    supportive_tools = len(set(evidence_results) & relevant_tools(task, config))
    required_tools = min_tools if min_tools is not None else route_min_tools(task, config)
    early_submit = len(evidence_results) < required_tools

    if normalized == expected:
        if early_submit:
            return 0.2
        if supportive_tools >= 1:
            support_bonus = min(0.2, 0.1 * max(0, supportive_tools - 1))
            return round(
                1.0
                + support_bonus
                + evidence_alignment_bonus(task, tool_results, normalized, config),
                3,
            )
        return -0.3

    if normalized == "escalate" and expected in {"reject", "request_changes"}:
        return -0.4 if early_submit else -0.2

    return -1.0 if early_submit else -0.8


def get_scoring_logic() -> dict[str, Any]:
    """Returns the constants and weights used for scoring, for UI transparency."""
    return {
        "tool_weights": TOOL_WEIGHTS,
        "relevant_weight_multiplier": 1.5,
        "base_step_reward": 0.1,
        "relevance_bonus": 0.2,
        "duplicate_penalty": -0.4,
        "post_min_tools_analysis_penalty": -0.3,
        "efficiency_decay_step": -0.05,
        "terminal_correct_evidence_base": 1.0,
        "terminal_correct_early_submit": 0.2,
        "terminal_correct_no_supportive_tool": -0.3,
        "terminal_escalate_high_risk": -0.2,
        "terminal_escalate_early_submit": -0.4,
        "terminal_wrong": -0.8,
        "terminal_wrong_early_submit": -1.0,
        "evidence_alignment_max": 0.25,
        "raw_range": [RAW_MIN, RAW_MAX],
    }


def _severity_counts(result: dict[str, Any]) -> dict[str, int]:
    counts = result.get("severity_counts")
    if isinstance(counts, dict):
        return {
            "critical": int(counts.get("critical", 0) or 0),
            "warning": int(counts.get("warning", 0) or 0),
        }

    findings = result.get("findings") or []
    if not isinstance(findings, list) or not findings:
        return {"critical": 0, "warning": 0}

    score = _tool_score(result)
    severity = "critical" if score <= 0.55 else "warning"
    return {
        "critical": len(findings) if severity == "critical" else 0,
        "warning": len(findings) if severity == "warning" else 0,
    }


def evidence_penalties(
    task: PRTask,
    tool_results: dict[str, dict[str, Any]],
    submitted_verdict: str,
) -> float:
    """Episode-end penalties for contradictions between evidence and verdict."""
    verdict = submitted_verdict.strip().lower()
    penalty = 0.0
    tool_results = _evidence_only(tool_results)
    severities = [_severity_counts(result) for result in tool_results.values()]
    total_critical = sum(item.get("critical", 0) for item in severities)
    total_warnings = sum(item.get("warning", 0) for item in severities)

    if total_critical > 0 and verdict == "approve":
        penalty -= 0.40
    if total_warnings > 3 and verdict == "approve":
        penalty -= 0.15
    if total_critical >= 2 and verdict == "request_changes":
        penalty -= 0.10

    all_clean = bool(tool_results) and all(_tool_score(result) > 0.85 for result in tool_results.values())
    if all_clean and verdict == "reject":
        penalty -= 0.20

    return round(penalty, 3)


def normalize_reward(raw: float) -> float:
    clipped = max(RAW_MIN, min(RAW_MAX, raw))
    return round(0.01 + (clipped - RAW_MIN) / (RAW_MAX - RAW_MIN) * 0.98, 4)


def outcome_summary(
    task: PRTask,
    tool_results: dict[str, dict[str, Any]],
    submitted_verdict: str | None = None,
    config: ReviewConfig | None = None,
) -> dict[str, Any]:
    """Returns: expected_verdict, relevant_tools (sorted list), aggregate_score, submitted_verdict."""
    return {
        "expected_verdict": task.expected_verdict,
        "relevant_tools": sorted(relevant_tools(task, config)),
        "aggregate_score": aggregate_tool_scores(tool_results, task, config),
        "submitted_verdict": submitted_verdict,
        "evidence_penalty": (
            evidence_penalties(task, tool_results, submitted_verdict)
            if submitted_verdict
            else 0.0
        ),
    }
