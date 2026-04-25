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


def route_requirements(tier: ModelTier) -> dict[str, int | bool]:
    if tier == ModelTier.SMALL:
        return {"min_tools": 1, "max_steps": 3, "requires_security": False}
    if tier == ModelTier.MEDIUM:
        return {"min_tools": 2, "max_steps": 5, "requires_security": True}
    return {"min_tools": 3, "max_steps": 7, "requires_security": True}


def prompt_route_context(
    obs: PRReviewObservation,
    config: ReviewConfig | dict[str, Any] | None = None,
) -> str:
    tier = select_tier(obs, config)
    requirements = route_requirements(tier)
    return (
        f"Route tier: {tier.value}\n"
        f"Minimum evidence tools: {requirements['min_tools']}\n"
        f"Maximum review steps: {requirements['max_steps']}\n"
        f"Security evidence required: {str(requirements['requires_security']).lower()}"
    )
