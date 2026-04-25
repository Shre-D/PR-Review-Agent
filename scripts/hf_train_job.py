# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface-hub>=1.0.0"]
# ///
"""HF Jobs entrypoint for PR Review GRPO training.

The job is intentionally small and clones the repository inside HF compute.
This avoids relying on the remote job having access to a local working tree.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

from huggingface_hub import HfApi


def run(command: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PR Review GRPO on Hugging Face Jobs.")
    parser.add_argument("--repo-url", default=os.getenv("REPO_URL", ""))
    parser.add_argument("--repo-ref", default=os.getenv("REPO_REF", "main"))
    parser.add_argument("--hub-model-id", default=os.getenv("HF_HUB_MODEL_ID", ""))
    parser.add_argument("--preset", default=os.getenv("PRESET", "a100"))
    parser.add_argument("--epochs", default=os.getenv("EPOCHS", "3"))
    parser.add_argument("--task-limit", default=os.getenv("TASK_LIMIT", "78"))
    parser.add_argument("--num-generations", default=os.getenv("NUM_GENERATIONS", "8"))
    parser.add_argument("--task-loader-mode", default=os.getenv("TASK_LOADER_MODE", "short"))
    parser.add_argument("--workdir", default=os.getenv("WORKDIR", "/tmp/pr-review-agent"))
    args = parser.parse_args()

    if not args.repo_url:
        raise SystemExit("Set REPO_URL to the git URL for this repository.")
    if not args.hub_model_id:
        raise SystemExit("Set HF_HUB_MODEL_ID to the model repo that should receive artifacts.")
    if not os.getenv("HF_TOKEN"):
        raise SystemExit("HF_TOKEN secret is required for Hub uploads.")

    repo_dir = Path(args.workdir)
    if repo_dir.exists():
        shutil.rmtree(repo_dir)

    run(["git", "clone", "--depth", "1", "--branch", args.repo_ref, args.repo_url, str(repo_dir)])
    run(["python", "-m", "pip", "install", "--upgrade", "pip"])
    run(["python", "-m", "pip", "install", "-e", ".[server,train,dev]", "matplotlib", "pandas"], cwd=repo_dir)
    run(["python", "tasks/build_task_bank.py", "--print-summary"], cwd=repo_dir)

    env = {**os.environ, "PR_REVIEW_TOOL_BACKEND": "heuristic"}
    output_dir = repo_dir / "grpo_checkpoint"
    rewards_dir = repo_dir / "rewards"
    rewards_dir.mkdir(exist_ok=True)

    run(
        [
            "python",
            "train/grpo_train.py",
            "--preset",
            args.preset,
            "--task-bank",
            "all",
            "--task-loader-mode",
            args.task_loader_mode,
            "--task-limit",
            args.task_limit,
            "--epochs",
            args.epochs,
            "--num-generations",
            args.num_generations,
            "--output-dir",
            str(output_dir),
            "--hub-model-id",
            args.hub_model_id,
        ],
        cwd=repo_dir,
        env=env,
    )
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
    run(
        [
            "python",
            "benchmarks/evaluate_trained_model.py",
            "--checkpoint",
            str(output_dir),
            "--preset",
            args.preset,
            "--task-bank",
            "all",
            "--task-loader-mode",
            args.task_loader_mode,
            "--output",
            str(rewards_dir / "trained_eval.json"),
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
    api.upload_file(
        repo_id=args.hub_model_id,
        repo_type="model",
        path_or_fileobj=str(output_dir / "training_log.csv"),
        path_in_repo="artifacts/training_log.csv",
    )
    print(f"Uploaded checkpoint and artifacts to https://huggingface.co/{args.hub_model_id}", flush=True)


if __name__ == "__main__":
    main()
