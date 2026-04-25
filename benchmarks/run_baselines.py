from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.pr_review_env.models import PRReviewAction
from envs.pr_review_env.server.pr_review_env import PRReviewEnv
from envs.pr_review_env.server.tasks import PRTask, load_tasks, task_review_config


def random_policy(observation, rng: random.Random) -> PRReviewAction:
    tool_name = rng.choice(
        [
            "check_security",
            "check_quality",
            "check_build_and_types",
            "check_tests",
            "check_config",
        ]
    )
    return PRReviewAction(tool_name=tool_name, arguments={"diff_str": observation.diff_str})


def heuristic_policy(observation) -> list[PRReviewAction]:
    actions: list[PRReviewAction] = []
    file_types = set(observation.changed_file_types)

    actions.append(
        PRReviewAction(tool_name="check_security", arguments={"diff_str": observation.diff_str})
    )
    if file_types & {"dockerfile", "yaml", "gitignore", "github_actions"}:
        actions.append(
            PRReviewAction(tool_name="check_config", arguments={"diff_str": observation.diff_str})
        )
    if observation.primary_language in {"python", "typescript", "javascript", "java", "rust"}:
        actions.append(
            PRReviewAction(tool_name="check_quality", arguments={"diff_str": observation.diff_str})
        )
    if observation.primary_language in {"java", "go", "rust"} or "build_manifest" in file_types or "github_actions" in file_types:
        actions.append(
            PRReviewAction(
                tool_name="check_build_and_types",
                arguments={"diff_str": observation.diff_str},
            )
        )
    if observation.primary_language in {"typescript", "javascript", "java", "python"} and not file_types <= {"yaml", "dockerfile", "gitignore", "github_actions"}:
        actions.append(
            PRReviewAction(tool_name="check_tests", arguments={"diff_str": observation.diff_str})
        )

    return actions


def decide_final_verdict(observation) -> PRReviewAction:
    scores = {name: payload.get("score", 1.0) for name, payload in observation.tool_results.items()}
    findings_count = {
        name: len(payload.get("findings") or [])
        for name, payload in observation.tool_results.items()
    }
    security_score = scores.get("check_security", 1.0)
    min_score = min(scores.values(), default=1.0)
    any_finding = any(count > 0 for count in findings_count.values())

    if security_score <= 0.55 or min_score <= 0.45:
        verdict = "reject"
    elif any_finding or min_score <= 0.85:
        verdict = "request_changes"
    else:
        verdict = "approve"

    return PRReviewAction(
        tool_name="submit_review",
        arguments={
            "verdict": verdict,
            "confidence": round(min(0.95, 0.55 + 0.1 * len(scores)), 2),
            "reasoning": f"Baseline verdict from tool scores: {json.dumps(scores, sort_keys=True)}",
        },
    )


def run_random(tasks: list[PRTask], task_path: str, loader_mode: str, episodes: int, seed: int) -> dict:
    rng = random.Random(seed)
    total_return = 0.0
    completed = 0
    for task in tasks[:episodes]:
        env = PRReviewEnv(
            seed=seed,
            task_path=task_path,
            review_config=task_review_config(task, mode=loader_mode),
        )
        observation = env.reset(task_id=task.task_id)
        episode_return = 0.0
        observation = env.step(random_policy(observation, rng))
        episode_return += observation.reward or 0.0
        observation = env.step(decide_final_verdict(observation))
        episode_return += observation.reward or 0.0
        total_return += episode_return
        completed += int(observation.done)
    return {
        "policy": "random",
        "episodes": episodes,
        "mean_episode_return": round(total_return / max(episodes, 1), 3),
        "completed": completed,
    }


def run_heuristic(tasks: list[PRTask], task_path: str, loader_mode: str, episodes: int) -> dict:
    total_return = 0.0
    completed = 0
    for task in tasks[:episodes]:
        env = PRReviewEnv(
            seed=7,
            task_path=task_path,
            review_config=task_review_config(task, mode=loader_mode),
        )
        observation = env.reset(task_id=task.task_id)
        episode_return = 0.0
        for action in heuristic_policy(observation):
            observation = env.step(action)
            episode_return += observation.reward or 0.0
        observation = env.step(decide_final_verdict(observation))
        episode_return += observation.reward or 0.0
        total_return += episode_return
        completed += int(observation.done)
    return {
        "policy": "heuristic",
        "episodes": episodes,
        "mean_episode_return": round(total_return / max(episodes, 1), 3),
        "completed": completed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run benchmark baselines for the PR review env.")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tasks-file", "--task-bank", dest="tasks_file", default="all")
    parser.add_argument(
        "--task-loader-mode",
        default="short",
        choices=["short", "empty", "full", "off", "none"],
    )
    args = parser.parse_args()

    tasks = load_tasks(args.tasks_file)
    episodes = min(args.episodes, len(tasks))
    print(json.dumps(run_random(tasks, args.tasks_file, args.task_loader_mode, episodes, args.seed)))
    print(json.dumps(run_heuristic(tasks, args.tasks_file, args.task_loader_mode, episodes)))


if __name__ == "__main__":
    main()
