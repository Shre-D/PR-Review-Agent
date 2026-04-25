from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.pr_review_env.models import PRReviewAction, PRReviewObservation
from envs.pr_review_env.server.context_loader import load_review_config
from envs.pr_review_env.server.pr_review_env import PRReviewEnv
from envs.pr_review_env.server.tasks import PRTask, load_tasks
from train.grpo_train import _action_with_state_args, build_obs_prompt, parse_action
from train.train_config import TrainingConfig


def checkpoint_ready(path: str | Path) -> bool:
    checkpoint = Path(path)
    if not checkpoint.exists():
        return False
    expected = {
        "adapter_config.json",
        "adapter_model.safetensors",
        "adapter_model.bin",
        "config.json",
        "model.safetensors",
        "pytorch_model.bin",
    }
    return any((checkpoint / name).exists() for name in expected)


def select_tasks(tasks_file: str, limit: int = 0) -> list[PRTask]:
    tasks = load_tasks(tasks_file)
    if limit and limit > 0:
        return tasks[:limit]
    return tasks


def build_prompt_messages(obs: PRReviewObservation, review_config: dict | None = None) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a code reviewer. Output exactly one JSON tool call and no prose.\n"
                "Tools: check_security, check_quality, check_build_and_types, check_tests, "
                "check_config, submit_review, escalate."
            ),
        },
        {"role": "user", "content": build_obs_prompt(obs, review_config)},
    ]


def generate_action_text(model, tokenizer, messages: list[dict[str, str]], max_new_tokens: int) -> str:
    import torch

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(next(model.parameters()).device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    new_tokens = output_ids[0][input_ids.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def load_policy_model(base_model: str, checkpoint: str):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    checkpoint_path = Path(checkpoint)
    tokenizer_source = checkpoint if (checkpoint_path / "tokenizer_config.json").exists() else base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if (checkpoint_path / "adapter_config.json").exists():
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            attn_implementation="eager",
        )
        model = PeftModel.from_pretrained(base, checkpoint)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            attn_implementation="eager",
        )

    model.eval()
    return model, tokenizer


def run_episode(
    model,
    tokenizer,
    task: PRTask,
    cfg: TrainingConfig,
    review_config: dict | None = None,
) -> dict[str, Any]:
    env = PRReviewEnv(seed=7, review_config=review_config)
    obs = env.reset(task_id=task.task_id)
    episode_return = 0.0
    last_action = None

    for _ in range(cfg.max_steps_per_episode):
        messages = build_prompt_messages(obs, review_config)
        response_text = generate_action_text(model, tokenizer, messages, cfg.max_new_tokens)
        action = _action_with_state_args(parse_action(response_text), obs)
        last_action = action
        obs = env.step(action)
        episode_return += obs.reward or 0.0
        if obs.done:
            break

    predicted = "incomplete"
    if last_action and last_action.tool_name == "submit_review":
        predicted = last_action.arguments.get("verdict", "incomplete")
    elif obs.final_verdict is not None:
        predicted = obs.final_verdict.verdict

    return {
        "task_id": task.task_id,
        "primary_language": task.primary_language,
        "changed_file_types": task.changed_file_types,
        "expected_verdict": task.expected_verdict,
        "predicted_verdict": predicted,
        "correct": predicted == task.expected_verdict,
        "episode_return": round(episode_return, 3),
        "tools_called": obs.tools_called,
        "done": obs.done,
    }


def summarize_results(results: list[dict[str, Any]], policy_name: str) -> dict[str, Any]:
    by_language = defaultdict(lambda: {"episodes": 0, "return_sum": 0.0, "correct": 0})
    total_return = 0.0
    correct = 0

    for result in results:
        total_return += result["episode_return"]
        correct += int(result["correct"])
        bucket = by_language[result["primary_language"]]
        bucket["episodes"] += 1
        bucket["return_sum"] += result["episode_return"]
        bucket["correct"] += int(result["correct"])

    episodes = len(results) or 1
    return {
        "policy": policy_name,
        "episodes": len(results),
        "accuracy": round(correct / episodes, 3),
        "mean_episode_return": round(total_return / episodes, 3),
        "by_language": {
            key: {
                "episodes": value["episodes"],
                "accuracy": round(value["correct"] / (value["episodes"] or 1), 3),
                "mean_episode_return": round(value["return_sum"] / (value["episodes"] or 1), 3),
            }
            for key, value in sorted(by_language.items())
        },
        "examples": results[:8],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained PR review SLM checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--tasks-file", default="tasks/tasks.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="0 means all tasks")
    parser.add_argument("--output", default=str(ROOT / "rewards" / "trained_eval.json"))
    parser.add_argument("--review-config", default="")
    parser.add_argument("--preset", default="cpu", choices=["cpu", "t4", "a100", "v100", "h100"])
    args = parser.parse_args()

    if not checkpoint_ready(args.checkpoint):
        raise SystemExit(
            f"Checkpoint is missing or incomplete: {args.checkpoint}\n"
            "Expected a LoRA adapter or full model file such as adapter_config.json, "
            "adapter_model.safetensors, config.json, or model.safetensors."
        )

    preset_map = {
        "cpu": TrainingConfig.for_cpu,
        "t4": TrainingConfig.for_t4,
        "a100": TrainingConfig.for_a100,
        "v100": TrainingConfig.for_v100,
        "h100": TrainingConfig.for_h100,
    }
    cfg = preset_map[args.preset]()
    review_config = load_review_config(args.review_config)
    model, tokenizer = load_policy_model(args.base_model, args.checkpoint)
    tasks = select_tasks(args.tasks_file, args.limit)

    results = [run_episode(model, tokenizer, task, cfg, review_config) for task in tasks]
    summary = summarize_results(results, policy_name="trained_slm")
    summary["checkpoint"] = args.checkpoint
    summary["base_model"] = args.base_model

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "episodes": summary["episodes"]}))


if __name__ == "__main__":
    main()
