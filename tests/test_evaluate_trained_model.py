from pathlib import Path

from benchmarks.evaluate_trained_model import (
    build_prompt_messages,
    checkpoint_ready,
    select_tasks,
    summarize_results,
)
from envs.pr_review_env.models import PRReviewObservation


def test_checkpoint_ready_detects_missing_and_adapter(tmp_path: Path):
    assert checkpoint_ready(tmp_path) is False

    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    assert checkpoint_ready(tmp_path) is True


def test_select_tasks_honors_limit():
    tasks = select_tasks("tasks/tasks.jsonl", limit=2)

    assert len(tasks) == 2


def test_build_prompt_messages_contains_observation_context():
    messages = build_prompt_messages(
        PRReviewObservation(
            primary_language="python",
            changed_file_types=["python"],
            repo_kind="service",
            pr_description="Test PR",
            diff_str="+x = 1",
        )
    )

    assert messages[0]["role"] == "system"
    assert "Test PR" in messages[1]["content"]
    assert "Route tier" in messages[1]["content"]


def test_summarize_results_computes_accuracy_and_return():
    summary = summarize_results(
        [
            {
                "task_id": "a",
                "primary_language": "python",
                "expected_verdict": "approve",
                "predicted_verdict": "approve",
                "correct": True,
                "episode_return": 1.0,
            },
            {
                "task_id": "b",
                "primary_language": "python",
                "expected_verdict": "reject",
                "predicted_verdict": "approve",
                "correct": False,
                "episode_return": -0.5,
            },
        ],
        policy_name="trained_slm",
    )

    assert summary["accuracy"] == 0.5
    assert summary["mean_episode_return"] == 0.25
    assert summary["by_language"]["python"]["episodes"] == 2
