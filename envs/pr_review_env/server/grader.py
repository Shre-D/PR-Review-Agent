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

RAW_MIN = -2.83
RAW_MAX = 1.51
RAW_FLOOR = RAW_MIN
RAW_NEAR_FLOOR = -2.55

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
) -> float:
    """Per-step reward for calling a tool (not submit_review/escalate).
    - submit_review/escalate -> 0.0
    - duplicate call -> 0.0 (no bonus, no negative — terminal handles correctness)
    - new call: base +0.05; +0.08 if in relevant_tools(task)
    - efficiency decay applied softly, but the step reward is clamped to >= 0.0
      so intermediate exploration never punishes the policy. Wrong outcomes
      are punished at submit time via terminal_reward.
    Rounded to 3 decimal places."""
    if tool_name in {"submit_review", "escalate"}:
        return 0.0

    if already_called:
        return RAW_NEAR_FLOOR

    reward = 0.05
    if tool_name in relevant_tools(task, config):
        domain_bonus = 0.08
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
        reward -= 0.03 * (step_count - decay_start)

    return round(max(0.0, reward), 3)


def terminal_reward(
    task: PRTask,
    submitted_verdict: str,
    tool_results: dict[str, dict[str, Any]],
    config: ReviewConfig | None = None,
    confidence: float | None = None,
) -> float:
    """Reward on submit_review or escalate.
    Correct + >=1 supportive tool -> 1.0 + evidence alignment bonus
    Correct + no supportive tools -> 0.35
    escalate when expected reject/request_changes -> 0.10
    Wrong verdict -> -0.55
    """
    normalized = submitted_verdict.strip().lower()
    expected = task.expected_verdict.strip().lower()
    supportive_tools = len(set(tool_results) & relevant_tools(task, config))
    min_tools = route_min_tools(task, config)
    early_submit = len(tool_results) < min_tools

    if normalized == expected:
        if early_submit:
            return -2.1
        if supportive_tools >= 1:
            return round(0.95 + evidence_alignment_bonus(task, tool_results, normalized, config), 3)
        return -1.45

    if normalized == "escalate" and expected in {"reject", "request_changes"}:
        return -0.8

    return -2.65


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
