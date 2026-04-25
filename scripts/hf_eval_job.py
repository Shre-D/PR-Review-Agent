# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface-hub>=1.0.0"]
# ///
"""HF Jobs entrypoint for evaluating an uploaded PR Review checkpoint."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


def run(command: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def upload_file_if_exists(api: HfApi, repo_id: str, source: Path, destination: str) -> None:
    if not source.exists():
        print(f"Skipping missing artifact: {source}", flush=True)
        return
    api.upload_file(
        repo_id=repo_id,
        repo_type="model",
        path_or_fileobj=str(source),
        path_in_repo=destination,
    )
    print(f"Uploaded artifact: {destination}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a PR Review checkpoint on HF Jobs.")
    parser.add_argument("--repo-url", default=os.getenv("REPO_URL", ""))
    parser.add_argument("--repo-ref", default=os.getenv("REPO_REF", "main"))
    parser.add_argument("--hub-model-id", default=os.getenv("HF_HUB_MODEL_ID", ""))
    parser.add_argument("--preset", default=os.getenv("PRESET", "t4"))
    parser.add_argument("--eval-limit", default=os.getenv("EVAL_LIMIT", "0"))
    parser.add_argument("--task-loader-mode", default=os.getenv("TASK_LOADER_MODE", "short"))
    parser.add_argument("--workdir", default=os.getenv("WORKDIR", "/tmp/pr-review-agent-eval"))
    args = parser.parse_args()

    if not args.repo_url:
        raise SystemExit("Set REPO_URL to the git URL for this repository.")
    if not args.hub_model_id:
        raise SystemExit("Set HF_HUB_MODEL_ID to the model repo containing the checkpoint.")
    if not os.getenv("HF_TOKEN"):
        raise SystemExit("HF_TOKEN secret is required for Hub uploads.")

    repo_dir = Path(args.workdir)
    if repo_dir.exists():
        shutil.rmtree(repo_dir)

    run(["git", "clone", "--depth", "1", "--branch", args.repo_ref, args.repo_url, str(repo_dir)])
    run(["uv", "pip", "install", "--python", sys.executable, "-e", ".", "--no-deps"], cwd=repo_dir)
    run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            sys.executable,
            "pydantic",
            "httpx",
            "fastapi",
            "uvicorn",
            "pyyaml",
            "trl",
            "peft",
            "bitsandbytes",
            "accelerate",
            "transformers",
            "datasets",
            "torch",
            "matplotlib",
            "pandas",
        ],
        cwd=repo_dir,
    )
    run(["python", "tasks/build_task_bank.py", "--print-summary"], cwd=repo_dir)

    checkpoint_dir = repo_dir / "uploaded_checkpoint"
    snapshot_download(
        repo_id=args.hub_model_id,
        repo_type="model",
        local_dir=checkpoint_dir,
    )

    env = {**os.environ, "PR_REVIEW_TOOL_BACKEND": "heuristic", "MPLCONFIGDIR": "/tmp/matplotlib"}
    rewards_dir = repo_dir / "rewards"
    rewards_dir.mkdir(exist_ok=True)

    run(
        [
            "python",
            "benchmarks/evaluate_baselines.py",
            "--task-bank",
            "all",
            "--task-loader-mode",
            args.task_loader_mode,
            "--output",
            str(rewards_dir / "baseline_eval.json"),
        ],
        cwd=repo_dir,
        env=env,
    )
    trained_eval_command = [
        "python",
        "benchmarks/evaluate_trained_model.py",
        "--checkpoint",
        str(checkpoint_dir),
        "--preset",
        args.preset,
        "--task-bank",
        "all",
        "--task-loader-mode",
        args.task_loader_mode,
        "--output",
        str(rewards_dir / "trained_eval.json"),
    ]
    if int(args.eval_limit) > 0:
        trained_eval_command.extend(["--limit", args.eval_limit])
    run(trained_eval_command, cwd=repo_dir, env=env)
    run(
        [
            "python",
            "benchmarks/generate_report.py",
            "--baseline",
            str(rewards_dir / "baseline_eval.json"),
            "--trained",
            str(rewards_dir / "trained_eval.json"),
            "--log",
            str(checkpoint_dir / "training_log.csv"),
            "--output",
            str(rewards_dir / "comparison_report.png"),
        ],
        cwd=repo_dir,
        env=env,
    )

    api = HfApi()
    api.upload_folder(
        repo_id=args.hub_model_id,
        repo_type="model",
        folder_path=str(rewards_dir),
        path_in_repo="artifacts/rewards",
    )
    upload_file_if_exists(api, args.hub_model_id, rewards_dir / "comparison_report.png", "artifacts/comparison_report.png")
    print(f"Uploaded eval artifacts to https://huggingface.co/{args.hub_model_id}", flush=True)


if __name__ == "__main__":
    main()
