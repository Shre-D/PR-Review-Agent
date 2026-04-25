import pytest

from envs.pr_review_env.server.grader import (
    aggregate_tool_scores,
    evidence_penalties,
    normalize_reward,
    relevant_tools,
    step_reward,
    terminal_reward,
    reward_floor,
)
from envs.pr_review_env.models import ReviewConfig
from envs.pr_review_env.server.tasks import PRTask


def _task(**overrides):
    payload = {
        "task_id": "sample",
        "diff_str": "diff --git a/app.py b/app.py\n+print('hi')\n",
        "pr_description": "sample",
        "primary_language": "python",
        "changed_file_types": ["python"],
        "repo_kind": "backend_service",
        "expected_verdict": "reject",
        "risk_domains": ["security"],
        "difficulty": "easy",
        "review_goal": "sample",
    }
    payload.update(overrides)
    return PRTask(**payload)


def test_relevant_tools_include_security_for_security_tasks():
    task = _task()
    assert "check_security" in relevant_tools(task)


def test_aggregate_tool_scores_renormalizes_present_tools():
    score = aggregate_tool_scores(
        {
            "check_security": {"score": 0.2},
            "check_quality": {"score": 0.8},
        }
    )
    assert 0.0 <= score <= 1.0


def test_step_reward_no_bonus_for_duplicate_calls():
    task = _task()
    assert step_reward(task, "check_security", already_called=True, step_count=2) < -2.0
    assert step_reward(task, "check_security", already_called=False, step_count=1) > 0.0


def test_terminal_reward_prefers_correct_verdict_with_evidence():
    task = _task()
    reward = terminal_reward(
        task,
        "reject",
        {"check_security": {"score": 0.1}, "check_quality": {"score": 0.3}},
    )
    assert reward > 0.9


def test_terminal_reward_penalizes_early_correct_submit():
    task = _task()
    reward = terminal_reward(
        task,
        "reject",
        {"check_security": {"score": 0.1}},
        min_tools=2,
    )

    assert reward < -2.4


def test_terminal_reward_rewards_extra_supportive_evidence():
    task = _task(risk_domains=["security", "quality"])
    one_tool = terminal_reward(
        task,
        "reject",
        {"check_security": {"score": 0.5}},
        min_tools=1,
    )
    two_tools = terminal_reward(
        task,
        "reject",
        {"check_security": {"score": 0.5}, "check_quality": {"score": 0.5}},
        min_tools=2,
    )

    assert two_tools > one_tool


def test_config_tool_weights_change_aggregate_score():
    score = aggregate_tool_scores(
        {
            "check_security": {"score": 0.0},
            "check_quality": {"score": 1.0},
        },
        ReviewConfig(tool_weights={"check_security": 0.9, "check_quality": 0.1}),
    )

    assert score < 0.5


def test_config_domain_priority_adjusts_step_reward():
    task = _task()
    default_reward = step_reward(task, "check_security", already_called=False, step_count=1)
    configured_reward = step_reward(
        task,
        "check_security",
        already_called=False,
        step_count=1,
        config=ReviewConfig(domain_priorities={"security": 2.0}),
    )

    assert configured_reward > default_reward


def test_terminal_reward_ignores_confidence_for_grader_independence():
    task = _task()
    low = terminal_reward(
        task,
        "reject",
        {"check_security": {"score": 0.1}},
        config=ReviewConfig(),
        confidence=0.0,
    )
    high = terminal_reward(
        task,
        "reject",
        {"check_security": {"score": 0.1}},
        config=ReviewConfig(),
        confidence=1.0,
    )

    assert high == low


def test_terminal_reward_uses_verdict_evidence_alignment():
    task = _task(expected_verdict="reject")

    reward = terminal_reward(
        task,
        "reject",
        {"check_security": {"score": 0.0}, "check_quality": {"score": 0.2}},
    )

    assert reward == pytest.approx(1.18)


def test_evidence_penalties_catches_critical_approve():
    task = _task(expected_verdict="approve")

    penalty = evidence_penalties(
        task,
        {"check_security": {"score": 0.1, "findings": ["SQL injection"]}},
        "approve",
    )

    assert penalty == -0.4


def test_normalize_reward_clips_to_spec_range():
    assert normalize_reward(-100.0) == 0.01
    assert normalize_reward(100.0) == 0.99
    assert reward_floor() == 0.01
