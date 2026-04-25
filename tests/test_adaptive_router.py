from envs.pr_review_env.models import AuthorContext, PRReviewObservation
from train.adaptive_router import ModelTier, prompt_route_context, route_requirements, select_tier


def test_select_tier_small_for_simple_diff():
    obs = PRReviewObservation(diff_str="+x = 1", changed_file_types=["python"])

    assert select_tier(obs) == ModelTier.SMALL


def test_select_tier_medium_for_multi_file_or_critical_path():
    obs = PRReviewObservation(
        diff_str="+x = 1",
        changed_file_types=["python", "yaml"],
        critical_paths_touched=["src/auth/"],
    )

    assert select_tier(obs) == ModelTier.MEDIUM


def test_select_tier_large_for_junior_on_critical_path():
    obs = PRReviewObservation(
        diff_str="+x = 1",
        changed_file_types=["python"],
        critical_paths_touched=["src/auth/"],
        author_context=AuthorContext(level="junior"),
    )

    assert select_tier(obs) == ModelTier.LARGE


def test_route_requirements_are_slm_only_process_constraints():
    requirements = route_requirements(ModelTier.LARGE)

    assert requirements["min_tools"] == 3
    assert requirements["max_steps"] == 7
    assert requirements["requires_security"] is True


def test_prompt_route_context_contains_no_model_escalation():
    obs = PRReviewObservation(diff_str="+x = 1", changed_file_types=["python"])
    context = prompt_route_context(obs)

    assert "Route tier: small" in context
    assert "7B" not in context
    assert "API" not in context
