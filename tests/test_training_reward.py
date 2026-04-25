from train.grpo_train import (
    _action_with_state_args,
    build_training_state_rows,
    make_env_reward_func,
    score_completion_locally,
)
from envs.pr_review_env.models import PRReviewAction, PRReviewObservation
from train.train_config import TrainingConfig


def test_score_completion_penalizes_malformed_output():
    reward = score_completion_locally("not json", "py_sql_injection")

    assert reward == 0.01


def test_action_with_state_args_adds_diff_for_analysis_tool():
    obs = PRReviewObservation(diff_str="+unsafe = True")
    action = _action_with_state_args(
        PRReviewAction(tool_name="check_security", arguments={}),
        obs,
    )

    assert action.arguments["diff_str"] == "+unsafe = True"


def test_score_completion_uses_live_environment_reward():
    reward = score_completion_locally(
        '{"tool_name":"check_security","arguments":{}}',
        "py_sql_injection",
    )

    assert reward > 0


def test_score_completion_replays_prior_state():
    # Replay two evidence tools so min_tools (=2 for SMALL tier) is satisfied
    # by real evidence, not by the (no-longer-stored) terminal call.
    replay = [
        {
            "type": "call_tool",
            "tool_name": "check_security",
            "arguments": {},
            "metadata": {},
        },
        {
            "type": "call_tool",
            "tool_name": "check_quality",
            "arguments": {},
            "metadata": {},
        },
    ]
    unsupported_reward = score_completion_locally(
        '{"tool_name":"submit_review","arguments":{"verdict":"reject","confidence":0.9,"reasoning":"security evidence"}}',
        "py_sql_injection",
    )
    reward = score_completion_locally(
        '{"tool_name":"submit_review","arguments":{"verdict":"reject","confidence":0.9,"reasoning":"security evidence"}}',
        "py_sql_injection",
        replay_actions=replay,
    )

    # score_completion_locally returns the normalized environment reward.
    # Evidence-backed submits should still beat unsupported correct verdicts.
    assert reward > unsupported_reward


def test_reward_func_scores_batch_with_task_ids():
    reward_func = make_env_reward_func()
    rewards = reward_func(
        completions=[
            '{"tool_name":"check_security","arguments":{}}',
            'bad output',
        ],
        task_id=["py_sql_injection", "py_sql_injection"],
        replay_actions=[[], []],
    )

    assert rewards[0] > 0
    assert rewards[1] == 0.01


def test_build_training_state_rows_contains_replayable_state():
    cfg = TrainingConfig.for_cpu()
    cfg.training_task_limit = 1
    rows = build_training_state_rows(cfg)

    assert rows
    assert {"prompt", "task_id", "replay_actions"} <= set(rows[0])


def test_cpu_training_config_uses_valid_grpo_generation_count():
    cfg = TrainingConfig.for_cpu()

    assert cfg.num_generations >= 2
