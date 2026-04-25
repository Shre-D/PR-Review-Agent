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
from envs.pr_review_env.server.tasks import load_tasks

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


def evaluate_policy(policy_name: str, seed: int) -> dict:
    env = PRReviewEnv(seed=seed)
    tasks = load_tasks()

    by_language = defaultdict(lambda: {"episodes": 0, "return_sum": 0.0, "correct": 0})
    by_file_type = defaultdict(lambda: {"episodes": 0, "return_sum": 0.0, "correct": 0})
    examples = []

    total_return = 0.0
    correct = 0

    for index, task in enumerate(tasks):
        observation = env.reset(task_id=task.task_id)
        episode_return = 0.0

        for action in _policy_actions(policy_name, observation, seed + index):
            observation = env.step(action)
            episode_return += observation.reward or 0.0

        final_action = decide_final_verdict(observation)
        observation = env.step(final_action)
        episode_return += observation.reward or 0.0

        verdict = final_action.arguments["verdict"]
        is_correct = int(verdict == task.expected_verdict)
        total_return += episode_return
        correct += is_correct

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
    return {
        "policy": policy_name,
        "episodes": len(tasks),
        "accuracy": round(correct / episodes, 3),
        "mean_episode_return": round(total_return / episodes, 3),
        "by_language": _finalize(by_language),
        "by_file_type": _finalize(by_file_type),
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the benchmark baselines across the full task bank.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output",
        default=str(ROOT / "rewards" / "baseline_eval.json"),
        help="Where to write the JSON summary.",
    )
    args = parser.parse_args()

    results = {
        "heuristic": evaluate_policy("heuristic", seed=args.seed),
        "random": evaluate_policy("random", seed=args.seed),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "policies": list(results)}))


if __name__ == "__main__":
    main()
