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
from envs.pr_review_env.server.tasks import PRTask, load_tasks, task_review_config
from train.adaptive_router import review_requirements
from train.grpo_train import SYSTEM_PROMPT, _action_with_state_args, parse_action, training_prompt
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


def select_tasks(
    tasks_file: str,
    limit: int = 0,
    holdout_file: str | None = None,
    mode: str = "all",
) -> list[PRTask]:
    """mode: 'all' | 'holdout_only' | 'train_only'. Holdout is the set of
    task_ids in `holdout_file` (one per line, '#' comments allowed)."""
    tasks = load_tasks(tasks_file)
    holdout: set[str] = set()
    if holdout_file:
        path = Path(holdout_file)
        if path.exists():
            holdout = {
                line.strip()
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.strip().startswith("#")
            }
    if mode == "holdout_only" and holdout:
        tasks = [task for task in tasks if task.task_id in holdout]
    elif mode == "train_only" and holdout:
        tasks = [task for task in tasks if task.task_id not in holdout]
    if limit and limit > 0:
        return tasks[:limit]
    return tasks


def build_prompt_messages(obs: PRReviewObservation, review_config: dict | None = None) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": training_prompt(obs, review_config, include_system=False)},
    ]


def generate_action_text(model, tokenizer, messages: list[dict[str, str]], max_new_tokens: int) -> str:
    import torch

    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    if hasattr(encoded, "to"):
        encoded = encoded.to(device)

    if hasattr(encoded, "keys") and "input_ids" in encoded.keys():
        model_inputs = {key: encoded[key] for key in encoded.keys()}
        input_ids = model_inputs["input_ids"]
    else:
        input_ids = encoded
        model_inputs = {"input_ids": input_ids}

    with torch.no_grad():
        output_ids = model.generate(
            **model_inputs,
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
    task_config = review_config or task_review_config(task, mode=cfg.task_loader_mode)
    env = PRReviewEnv(seed=7, task_path=cfg.tasks_file, review_config=task_config)
    obs = env.reset(task_id=task.task_id)
    route = review_requirements(obs, task_config)
    episode_return = 0.0
    last_action = None
    invalid_actions = 0

    for _ in range(route["max_steps"]):
        messages = build_prompt_messages(obs, task_config)
        response_text = generate_action_text(model, tokenizer, messages, cfg.max_new_tokens)
        parsed = parse_action(response_text)
        action = _action_with_state_args(parsed, obs)
        if parsed.tool_name not in set(obs.available_tools):
            invalid_actions += 1
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
        "route_tier": route["tier"],
        "route_min_tools": route["min_tools"],
        "route_max_steps": route["max_steps"],
        "steps_used": obs.step_count,
        "duplicate_tools": max(
            0,
            len([t for t in obs.tools_called if t not in {"submit_review", "escalate"}])
            - len(set(t for t in obs.tools_called if t not in {"submit_review", "escalate"})),
        ),
        "invalid_actions": invalid_actions,
        "early_submit": bool(
            obs.final_verdict is not None
            and len({k for k in obs.tool_results if k not in {"submit_review", "escalate"}})
            < int(route["min_tools"])
        ),
        "over_budget": not obs.done,
        "evidence_backed_verdict": bool(
            obs.final_verdict is not None
            and len(
                {k for k in obs.tool_results if k not in {"submit_review", "escalate"}}
                & set(obs.metadata.get("relevant_tools", []))
            )
            > 0
        ),
        "context_dependent": bool(getattr(task, "context_requirements", [])),
    }


def summarize_results(results: list[dict[str, Any]], policy_name: str) -> dict[str, Any]:
    by_language = defaultdict(lambda: {"episodes": 0, "return_sum": 0.0, "correct": 0})
    total_return = 0.0
    correct = 0
    duplicate_total = 0
    invalid_total = 0
    early_submit_total = 0
    over_budget_total = 0
    evidence_backed_total = 0
    steps_by_tier = defaultdict(list)
    tool_calls = []
    unique_tool_calls = []
    context_results = []

    for result in results:
        total_return += result["episode_return"]
        correct += int(result["correct"])
        bucket = by_language[result["primary_language"]]
        bucket["episodes"] += 1
        bucket["return_sum"] += result["episode_return"]
        bucket["correct"] += int(result["correct"])
        duplicate_total += result.get("duplicate_tools", 0)
        invalid_total += result.get("invalid_actions", 0)
        early_submit_total += int(result.get("early_submit", False))
        over_budget_total += int(result.get("over_budget", False))
        evidence_backed_total += int(result.get("evidence_backed_verdict", False))
        steps_by_tier[result.get("route_tier", "unknown")].append(result.get("steps_used", 0))
        non_terminal = [t for t in result.get("tools_called", []) if t not in {"submit_review", "escalate"}]
        tool_calls.append(len(non_terminal))
        unique_tool_calls.append(len(set(non_terminal)))
        if result.get("context_dependent"):
            context_results.append(result)

    episodes = len(results) or 1
    summary = {
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
        "duplicate_tool_rate": round(duplicate_total / episodes, 3),
        "invalid_action_rate": round(invalid_total / episodes, 3),
        "early_submit_rate": round(early_submit_total / episodes, 3),
        "over_budget_rate": round(over_budget_total / episodes, 3),
        "evidence_backed_verdict_rate": round(evidence_backed_total / episodes, 3),
        "mean_steps_by_tier": {
            tier: round(sum(values) / max(len(values), 1), 3)
            for tier, values in sorted(steps_by_tier.items())
        },
        "mean_tool_calls": round(sum(tool_calls) / max(len(tool_calls), 1), 3),
        "mean_unique_tool_calls": round(sum(unique_tool_calls) / max(len(unique_tool_calls), 1), 3),
    }
    if context_results:
        context_episodes = len(context_results)
        summary["context_dependent"] = {
            "episodes": context_episodes,
            "accuracy": round(sum(int(item["correct"]) for item in context_results) / context_episodes, 3),
            "mean_episode_return": round(sum(item["episode_return"] for item in context_results) / context_episodes, 3),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained PR review SLM checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--tasks-file", "--task-bank", dest="tasks_file", default="all")
    parser.add_argument("--limit", type=int, default=0, help="0 means all tasks")
    parser.add_argument("--output", default=str(ROOT / "rewards" / "trained_eval.json"))
    parser.add_argument("--review-config", default="")
    parser.add_argument("--preset", default="cpu", choices=["cpu", "t4", "a100", "v100", "h100"])
    parser.add_argument(
        "--task-loader-mode",
        default="short",
        choices=["short", "empty", "full", "off", "none"],
    )
    parser.add_argument(
        "--holdout-file",
        default=str(ROOT / "tasks" / "holdout_ids.txt"),
        help="Holdout task_id list. Used with --eval-mode.",
    )
    parser.add_argument(
        "--eval-mode",
        default="all",
        choices=["all", "holdout_only", "train_only"],
        help="Slice of the bank to evaluate. 'all' = full bank; 'holdout_only' = unseen tasks; 'train_only' = excludes holdout.",
    )
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
    cfg.tasks_file = args.tasks_file
    cfg.task_loader_mode = args.task_loader_mode
    review_config = load_review_config(args.review_config)
    model, tokenizer = load_policy_model(args.base_model, args.checkpoint)
    tasks = select_tasks(args.tasks_file, args.limit, args.holdout_file, args.eval_mode)

    results = [run_episode(model, tokenizer, task, cfg, review_config) for task in tasks]
    summary = summarize_results(results, policy_name=f"trained_slm_{args.eval_mode}")
    summary["checkpoint"] = args.checkpoint
    summary["base_model"] = args.base_model
    summary["eval_mode"] = args.eval_mode

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "episodes": summary["episodes"]}))


if __name__ == "__main__":
    main()
