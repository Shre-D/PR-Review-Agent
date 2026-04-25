from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.run_baselines import decide_final_verdict, heuristic_policy
from envs.pr_review_env.models import PRReviewAction, PRReviewObservation
from envs.pr_review_env.server.pr_review_env import PRReviewEnv
from envs.pr_review_env.server.tasks import load_tasks


def build_prompt(observation: PRReviewObservation) -> str:
    return "\n".join(
        [
            "You are reviewing a pull request diff.",
            f"Primary language: {observation.primary_language}",
            f"Changed file types: {', '.join(observation.changed_file_types)}",
            f"Repository kind: {observation.repo_kind}",
            "Available tools: " + ", ".join(observation.available_tools),
            "",
            "PR description:",
            observation.pr_description,
            "",
            "Diff:",
            observation.diff_str,
            "",
            "Respond with the next tool call or the final verdict.",
        ]
    )


def action_to_message(action: PRReviewAction) -> str:
    if action.tool_name == "submit_review":
        return json.dumps(
            {
                "tool_name": action.tool_name,
                "arguments": {
                    "verdict": action.arguments["verdict"],
                    "reasoning": action.arguments["reasoning"],
                },
            },
            sort_keys=True,
        )

    return json.dumps(
        {
            "tool_name": action.tool_name,
            "arguments": {"diff_str": "<diff omitted for compactness>"},
        },
        sort_keys=True,
    )


def export_teacher_traces(limit: int | None = None) -> list[dict]:
    env = PRReviewEnv(seed=7)
    tasks = load_tasks()
    if limit is not None:
        tasks = tasks[:limit]

    traces: list[dict] = []
    for task in tasks:
        observation = env.reset(task_id=task.task_id)
        prompt = build_prompt(observation)
        actions = heuristic_policy(observation)

        step_trace = []
        for action in actions:
            observation = env.step(action)
            step_trace.append(
                {
                    "action": action.model_dump(),
                    "reward": observation.reward,
                    "last_tool_result": observation.last_tool_result,
                }
            )

        final_action = decide_final_verdict(observation)
        observation = env.step(final_action)
        step_trace.append(
            {
                "action": final_action.model_dump(),
                "reward": observation.reward,
                "last_tool_result": observation.last_tool_result,
            }
        )

        traces.append(
            {
                "task_id": task.task_id,
                "prompt": prompt,
                "target_response": action_to_message(final_action),
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": action_to_message(final_action)},
                ],
                "trajectory": step_trace,
                "expected_verdict": task.expected_verdict,
                "predicted_verdict": final_action.arguments["verdict"],
                "episode_return": round(sum(item["reward"] or 0.0 for item in step_trace), 3),
            }
        )

    return traces


def main() -> None:
    parser = argparse.ArgumentParser(description="Export heuristic teacher traces for prompt tuning or offline analysis.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--output",
        default=str(ROOT / "train" / "teacher_traces.jsonl"),
        help="Output JSONL file.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    traces = export_teacher_traces(limit=args.limit)
    with output_path.open("w", encoding="utf-8") as handle:
        for trace in traces:
            handle.write(json.dumps(trace) + "\n")
    print(json.dumps({"output": str(output_path), "traces": len(traces)}))


if __name__ == "__main__":
    main()
