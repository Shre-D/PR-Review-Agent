from __future__ import annotations

from enum import Enum
from typing import Any

from envs.pr_review_env.models import PRReviewObservation, ReviewConfig


class ModelTier(str, Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


def _added_line_count(diff_str: str) -> int:
    return sum(
        1
        for line in diff_str.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )


def select_tier(
    obs: PRReviewObservation,
    config: ReviewConfig | dict[str, Any] | None = None,
) -> ModelTier:
    loc_added = _added_line_count(obs.diff_str)
    changed_kinds = set(obs.changed_file_types)
    is_critical = bool(obs.critical_paths_touched)
    is_junior = obs.author_context.level in {"junior", "non_tech"}

    if config is not None and not isinstance(config, ReviewConfig):
        config = ReviewConfig.model_validate(config)

    if is_critical and is_junior:
        return ModelTier.LARGE
    if loc_added > 800 or (len(changed_kinds) > 2 and loc_added > 300):
        return ModelTier.LARGE
    if loc_added > 200 or len(changed_kinds) > 1 or is_critical:
        return ModelTier.MEDIUM
    return ModelTier.SMALL


_DEFAULT_AUTHOR_DEPTH = {
    "junior":   1.5,
    "mid":      1.0,
    "senior":   0.75,
    "lead":     0.6,
    "non_tech": 1.3,
    "unknown":  1.0,
}


def _clamp_depth_for_tier(tier: ModelTier, depth: float) -> float:
    # Keep author-depth scaling bounded so route budgets stay intentional.
    if tier == ModelTier.SMALL:
        return max(0.8, min(1.2, depth))
    if tier == ModelTier.MEDIUM:
        return max(0.75, min(1.3, depth))
    return max(0.7, min(1.35, depth))


def route_requirements(
    tier: ModelTier,
    obs: PRReviewObservation | None = None,
    config: ReviewConfig | dict[str, Any] | None = None,
) -> dict[str, int | bool]:
    if tier == ModelTier.SMALL:
        base = {"min_tools": 2, "max_steps": 5, "requires_security": False}
    elif tier == ModelTier.MEDIUM:
        base = {"min_tools": 3, "max_steps": 7, "requires_security": True}
    else:
        base = {"min_tools": 4, "max_steps": 9, "requires_security": True}

    if obs is None:
        return base

    if config is not None and not isinstance(config, ReviewConfig):
        config = ReviewConfig.model_validate(config)
    depth_table = (
        config.author_depth if config and config.author_depth else _DEFAULT_AUTHOR_DEPTH
    )
    depth = float(depth_table.get(obs.author_context.level, 1.0))
    depth = _clamp_depth_for_tier(tier, depth)
    base["max_steps"] = max(2, round(base["max_steps"] * depth))
    base["min_tools"] = max(1, min(base["max_steps"], round(base["min_tools"] * depth)))
    return base


def review_requirements(
    obs: PRReviewObservation,
    config: ReviewConfig | dict[str, Any] | None = None,
) -> dict[str, Any]:
    tier = select_tier(obs, config)
    requirements = route_requirements(tier, obs, config)
    evidence_tools_called = [
        tool
        for tool in obs.tools_called
        if tool not in {"submit_review", "escalate"}
    ]
    relevant_tools = set(
        obs.metadata.get("relevant_tools", [])
        if isinstance(obs.metadata, dict)
        else []
    )
    relevant_tools_called = [tool for tool in evidence_tools_called if tool in relevant_tools]
    return {
        "tier": tier.value,
        "min_tools": int(requirements["min_tools"]),
        "max_steps": int(requirements["max_steps"]),
        "requires_security": bool(requirements["requires_security"]),
        "remaining_steps": max(0, int(requirements["max_steps"]) - int(obs.step_count)),
        "evidence_tools_called": evidence_tools_called,
        "relevant_tools_called": relevant_tools_called,
    }


def prompt_route_context(
    obs: PRReviewObservation,
    config: ReviewConfig | dict[str, Any] | None = None,
) -> str:
    requirements = review_requirements(obs, config)
    return (
        f"Route tier: {requirements['tier']}\n"
        f"Author level: {obs.author_context.level}\n"
        f"Minimum evidence tools: {requirements['min_tools']}\n"
        f"Maximum review steps: {requirements['max_steps']}\n"
        f"Remaining review steps: {requirements['remaining_steps']}\n"
        f"Security evidence required: {str(requirements['requires_security']).lower()}"
    )
