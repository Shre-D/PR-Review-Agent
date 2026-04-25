import os

from envs.pr_review_env.models import PRReviewAction
from envs.pr_review_env.models import ReviewConfig
from envs.pr_review_env.server.pr_review_env import PRReviewEnv


def test_episode_smoke(monkeypatch):
    monkeypatch.setenv("PR_REVIEW_TOOL_BACKEND", "heuristic")

    env = PRReviewEnv(seed=7)
    observation = env.reset(task_id="py_sql_injection")

    security = env.step(
        PRReviewAction(
            tool_name="check_security",
            arguments={"diff_str": observation.diff_str},
        )
    )
    assert security.last_tool_name == "check_security"
    assert "check_security" in security.tool_results

    final = env.step(
        PRReviewAction(
            tool_name="submit_review",
            arguments={
                "verdict": "reject",
                "confidence": 0.9,
                "reasoning": "Security scanner evidence indicates injection risk.",
            },
        )
    )
    assert final.done is True
    assert final.reward is not None
    assert final.last_tool_result["confidence"] == 0.9


def test_env_injects_task_id_for_fixture_backed_tools():
    env = PRReviewEnv(seed=7)
    observation = env.reset(task_id="yaml_prod_debug")

    config = env.step(
        PRReviewAction(
            tool_name="check_config",
            arguments={"diff_str": observation.diff_str},
        )
    )

    assert config.last_tool_result["task_id"] == "yaml_prod_debug"
    assert config.last_tool_result["analysis_mode"] == "fixture_backed"
    assert "pyyaml" in config.last_tool_result["real_tools_used"]


def test_env_populates_config_aware_observation_and_final_verdict(monkeypatch):
    monkeypatch.setenv("PR_REVIEW_TOOL_BACKEND", "heuristic")
    env = PRReviewEnv(
        seed=7,
        review_config=ReviewConfig(critical_paths=["app/search.py"]),
    )
    observation = env.reset(task_id="py_sql_injection")

    assert observation.author_context.level == "mid"
    assert observation.critical_paths_touched == ["app/search.py"]
    assert observation.estimated_risk_level == "high"

    observation = env.step(
        PRReviewAction(
            tool_name="check_security",
            arguments={"diff_str": observation.diff_str},
        )
    )
    final = env.step(
        PRReviewAction(
            tool_name="submit_review",
            arguments={
                "verdict": "reject",
                "confidence": 0.9,
                "reasoning": "Security evidence indicates injection risk.",
            },
        )
    )

    assert final.final_verdict is not None
    assert final.final_verdict.verdict == "reject"
    assert final.final_verdict.critical_findings
