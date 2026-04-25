from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.pr_review_env.models import PRReviewAction
from envs.pr_review_env.server.pr_review_env import PRReviewEnv
from envs.pr_review_env.server.tasks import load_tasks, task_review_config
from train.adaptive_router import review_requirements

from benchmarks.run_baselines import decide_final_verdict, heuristic_policy, random_policy


PolicyFn = Callable[[object], list[PRReviewAction] | PRReviewAction]


def _policy_actions(name: str, observation, seed: int) -> list[PRReviewAction]:
    if name == "heuristic":
        return heuristic_policy(observation)
    if name == "random":
        import random

        stable_offset = int(hashlib.sha256(observation.task_id.encode("utf-8")).hexdigest()[:8], 16)
        return [random_policy(observation, random.Random(seed + stable_offset % 997))]
    raise ValueError(f"Unknown policy: {name}")


def evaluate_policy(
    policy_name: str,
    seed: int,
    tasks_file: str = "all",
    task_loader_mode: str = "short",
) -> dict:
    tasks = load_tasks(tasks_file)

    by_language = defaultdict(lambda: {"episodes": 0, "return_sum": 0.0, "correct": 0})
    by_file_type = defaultdict(lambda: {"episodes": 0, "return_sum": 0.0, "correct": 0})
    examples = []

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

    for index, task in enumerate(tasks):
        review_config = task_review_config(task, mode=task_loader_mode)
        env = PRReviewEnv(seed=seed, task_path=tasks_file, review_config=review_config)
        observation = env.reset(task_id=task.task_id)
        route = review_requirements(observation, review_config)
        episode_return = 0.0
        invalid_actions = 0

        for action in _policy_actions(policy_name, observation, seed + index):
            if action.tool_name not in set(observation.available_tools):
                invalid_actions += 1
            observation = env.step(action)
            episode_return += observation.reward or 0.0

        final_action = decide_final_verdict(observation)
        if final_action.tool_name not in set(observation.available_tools):
            invalid_actions += 1
        observation = env.step(final_action)
        episode_return += observation.reward or 0.0

        verdict = final_action.arguments["verdict"]
        is_correct = int(verdict == task.expected_verdict)
        total_return += episode_return
        correct += is_correct
        duplicate_total += max(
            0,
            len([t for t in observation.tools_called if t not in {"submit_review", "escalate"}])
            - len(set(t for t in observation.tools_called if t not in {"submit_review", "escalate"})),
        )
        invalid_total += invalid_actions
        evidence_tool_results = {
            tool: result
            for tool, result in observation.tool_results.items()
            if tool not in {"submit_review", "escalate"}
        }
        early_submit_total += int(len(evidence_tool_results) < int(route["min_tools"]))
        over_budget_total += int(observation.step_count > int(route["max_steps"]))
        evidence_backed_total += int(
            len(set(evidence_tool_results) & set(observation.metadata.get("relevant_tools", []))) > 0
        )
        steps_by_tier[route["tier"]].append(observation.step_count)
        non_terminal = [t for t in observation.tools_called if t not in {"submit_review", "escalate"}]
        tool_calls.append(len(non_terminal))
        unique_tool_calls.append(len(set(non_terminal)))
        if getattr(task, "context_requirements", []):
            context_results.append({"correct": is_correct, "episode_return": episode_return})

        by_language[task.primary_language]["episodes"] += 1
        by_language[task.primary_language]["return_sum"] += episode_return
        by_language[task.primary_language]["correct"] += is_correct

        for file_type in task.changed_file_types:
            by_file_type[file_type]["episodes"] += 1
            by_file_type[file_type]["return_sum"] += episode_return
            by_file_type[file_type]["correct"] += is_correct

        if len(examples) < 8:
            examples.append(
                {
                    "task_id": task.task_id,
                    "primary_language": task.primary_language,
                    "changed_file_types": task.changed_file_types,
                    "expected_verdict": task.expected_verdict,
                    "predicted_verdict": verdict,
                    "episode_return": round(episode_return, 3),
                    "tools_called": observation.tools_called,
                }
            )

    def _finalize(bucket: dict[str, dict]) -> dict[str, dict]:
        finalized = {}
        for key, value in sorted(bucket.items()):
            episodes = value["episodes"] or 1
            finalized[key] = {
                "episodes": value["episodes"],
                "accuracy": round(value["correct"] / episodes, 3),
                "mean_episode_return": round(value["return_sum"] / episodes, 3),
            }
        return finalized

    episodes = len(tasks) or 1
    summary = {
        "policy": policy_name,
        "episodes": len(tasks),
        "accuracy": round(correct / episodes, 3),
        "mean_episode_return": round(total_return / episodes, 3),
        "by_language": _finalize(by_language),
        "by_file_type": _finalize(by_file_type),
        "examples": examples,
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
            "accuracy": round(sum(item["correct"] for item in context_results) / context_episodes, 3),
            "mean_episode_return": round(
                sum(item["episode_return"] for item in context_results) / context_episodes, 3
            ),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the benchmark baselines across the full task bank.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output",
        default=str(ROOT / "rewards" / "baseline_eval.json"),
        help="Where to write the JSON summary.",
    )
    parser.add_argument("--tasks-file", "--task-bank", dest="tasks_file", default="all")
    parser.add_argument(
        "--task-loader-mode",
        default="short",
        choices=["short", "empty", "full", "off", "none"],
    )
    args = parser.parse_args()

    results = {
        "heuristic": evaluate_policy(
            "heuristic",
            seed=args.seed,
            tasks_file=args.tasks_file,
            task_loader_mode=args.task_loader_mode,
        ),
        "random": evaluate_policy(
            "random",
            seed=args.seed,
            tasks_file=args.tasks_file,
            task_loader_mode=args.task_loader_mode,
        ),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "policies": list(results)}))


if __name__ == "__main__":
    main()
