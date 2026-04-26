from train.grpo_train import (
    _action_with_state_args,
    _clean_review_config_payload,
    _tokenize_sft_examples,
    build_training_state_rows,
    make_env_reward_func,
    score_completion_locally,
    training_prompt,
)
from envs.pr_review_env.models import PRReviewAction, PRReviewObservation
from envs.pr_review_env.server.grader import RAW_FLOOR
from train.train_config import TrainingConfig


def test_score_completion_penalizes_malformed_output():
    reward = score_completion_locally("not json", "py_sql_injection")

    assert reward == RAW_FLOOR


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

    # score_completion_locally returns raw rewards.
    # Evidence-backed submits should beat unsupported correct verdicts.
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
    assert rewards[1] == RAW_FLOOR


def test_build_training_state_rows_contains_replayable_state():
    cfg = TrainingConfig.for_cpu()
    cfg.training_task_limit = 1
    rows = build_training_state_rows(cfg)

    assert rows
    assert {"prompt", "task_id", "replay_actions", "phase"} <= set(rows[0])


def test_training_prompt_switches_to_terminal_ready_after_min_tools():
    obs = PRReviewObservation(
        diff_str="+unsafe = True",
        primary_language="python",
        changed_file_types=["python"],
        repo_kind="backend_service",
        tool_results={
            "check_security": {"score": 0.1},
            "check_quality": {"score": 0.8},
        },
        metadata={"route": {"min_tools": 2}, "relevant_tools": ["check_security"]},
    )

    prompt = training_prompt(obs)

    assert "Review phase: TERMINAL_READY" in prompt
    assert "Submit the final verdict now" in prompt


def test_sft_tokenization_masks_prompt_and_keeps_target():
    class TinyTokenizer:
        eos_token_id = 0

        def __call__(self, text, add_special_tokens=False):  # noqa: ANN001
            return {"input_ids": [ord(char) for char in text]}

    rows = [
        {
            "prompt": "prompt text",
            "target_response": '{"tool_name":"submit_review","arguments":{}}',
            "text": "",
        }
    ]

    item = _tokenize_sft_examples(TinyTokenizer(), rows, max_length=256)[0]

    first_label = next(label for label in item["labels"] if label != -100)
    assert first_label == ord("{")
    assert item["labels"][-1] == 0


def test_terminal_ready_rows_are_oversampled():
    cfg = TrainingConfig.for_cpu()
    cfg.training_task_limit = 1
    rows = build_training_state_rows(cfg, terminal_multiplier=3)

    phases = [row["phase"] for row in rows]
    assert "terminal_ready" in phases
    assert phases.count("terminal_ready") >= 3


def test_cpu_training_config_uses_valid_grpo_generation_count():
    cfg = TrainingConfig.for_cpu()

    assert cfg.num_generations >= 2


def test_clean_review_config_payload_removes_dataset_nulls():
    payload = {
        "domain_priorities": {"security": 1.6, "quality": None},
        "tool_weights": {"check_security": None},
        "author_depth": {"junior": None, "mid": 1.0},
        "enabled_tools": ["ruff", None],
    }

    cleaned = _clean_review_config_payload(payload)

    assert cleaned["domain_priorities"] == {"security": 1.6}
    assert cleaned["tool_weights"] == {}
    assert cleaned["author_depth"] == {"mid": 1.0}
    assert cleaned["enabled_tools"] == ["ruff"]
