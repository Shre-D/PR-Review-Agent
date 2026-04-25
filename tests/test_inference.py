import json
import os
import subprocess
import sys

from benchmarks.run_baselines import heuristic_policy
from envs.pr_review_env.models import PRReviewAction
from envs.pr_review_env.server.pr_review_env import PRReviewEnv
from envs.pr_review_env.server.tasks import get_task_by_id, task_review_config
from inference import build_review_packet
from train.grpo_train import _action_with_state_args


def test_heuristic_plan_actions_are_consumed_without_repeating_first_tool():
    task = get_task_by_id("docker_root_user")
    env = PRReviewEnv(review_config=task_review_config(task, mode="short"))
    obs = env.reset(task_id=task.task_id)
    planned = heuristic_policy(obs)
    called = []
    for planned_action in planned:
        action = _action_with_state_args(planned_action, obs)
        called.append(action.tool_name)
        obs = env.step(action)
        if obs.done:
            break

    assert called == [action.tool_name for action in planned[: len(called)]]
    assert len(called) == len(set(called))


def test_build_review_packet_hides_expected_by_default():
    task = get_task_by_id("py_sql_injection")
    env = PRReviewEnv(review_config=task_review_config(task, mode="short"))
    obs = env.reset(task_id=task.task_id)
    obs = env.step(_action_with_state_args(heuristic_policy(obs)[0], obs))
    obs = env.step(
        _action_with_state_args(
            PRReviewAction(
                tool_name="submit_review",
                arguments={"verdict": "reject", "confidence": 0.9, "reasoning": "security evidence"},
            ),
            obs,
        )
    )

    packet = build_review_packet(
        task,
        obs,
        episode_return=env.state.cumulative_reward,
        model_name="heuristic",
        steps_used=obs.step_count,
        route={"tier": "medium", "min_tools": 3, "max_steps": 7},
        include_expected=False,
    )
    assert "expected_verdict" not in packet
    assert "correct" not in packet
    assert {"task_id", "predicted_verdict", "route", "tool_cost", "reward"} <= set(packet)


def _run_inference(*args: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "PR_REVIEW_TOOL_BACKEND": "heuristic"}
    return subprocess.run(
        [sys.executable, "inference.py", "--model", "heuristic", "--limit", "1", *args],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )


def test_inference_trace_output_is_unambiguous_and_can_hide_diff():
    result = _run_inference("--output-format", "trace", "--hide-diff")

    assert "[END] correct=" in result.stdout
    assert "score=" not in result.stdout
    assert '"diff_str": "<hidden>"' in result.stdout
    assert "SELECT * FROM users" not in result.stdout


def test_inference_packet_output_is_human_readable_without_benchmark_fields():
    result = _run_inference("--output-format", "packet")

    assert "PR: py_sql_injection" in result.stdout
    assert "Decision: reject" in result.stdout
    assert "correct" not in result.stdout
    assert "expected_verdict" not in result.stdout


def test_inference_jsonl_output_is_valid_packet_without_expected_by_default():
    result = _run_inference("--output-format", "jsonl")
    packet = json.loads(result.stdout)

    assert packet["task_id"] == "py_sql_injection"
    assert packet["predicted_verdict"] == "reject"
    assert "expected_verdict" not in packet
    assert "correct" not in packet
