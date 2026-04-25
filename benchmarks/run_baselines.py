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
    security_score = scores.get("check_security", 1.0)
    config_score = scores.get("check_config", 1.0)
    quality_score = scores.get("check_quality", 1.0)
    build_score = scores.get("check_build_and_types", 1.0)
    tests_score = scores.get("check_tests", 1.0)
    min_score = min(scores.values(), default=1.0)

    if security_score <= 0.55:
        verdict = "reject"
    elif min_score <= 0.74 or config_score <= 0.72 or quality_score <= 0.68 or build_score <= 0.7 or tests_score <= 0.65:
        verdict = "request_changes"
    else:
        verdict = "approve"

    return PRReviewAction(
        tool_name="submit_review",
        arguments={
            "verdict": verdict,
            "reasoning": f"Baseline verdict from tool scores: {json.dumps(scores, sort_keys=True)}",
        },
    )


def run_random(env: PRReviewEnv, episodes: int, seed: int) -> dict:
    rng = random.Random(seed)
    total_return = 0.0
    completed = 0
    for _ in range(episodes):
        observation = env.reset()
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


def run_heuristic(env: PRReviewEnv, episodes: int) -> dict:
    total_return = 0.0
    completed = 0
    for _ in range(episodes):
        observation = env.reset()
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
    args = parser.parse_args()

    env = PRReviewEnv(seed=args.seed)
    print(json.dumps(run_random(env, args.episodes, args.seed)))
    print(json.dumps(run_heuristic(env, args.episodes)))


if __name__ == "__main__":
    main()
