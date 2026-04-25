"""GRPO training for the PR review environment.

SLM-only: no LLM calls during training. Uses QLoRA for fast fine-tuning
on consumer/free-tier GPUs (T4 16GB, A100 40GB).

Training is SLM-only. GRPO completions are scored by replaying benchmark state
inside the local PRReviewEnv and returning the environment reward for the
generated tool call.

Usage:
    # Auto-detect GPU and pick preset:
    python train/grpo_train.py --preset t4 --env-url http://localhost:8000

    # Explicit options:
    python train/grpo_train.py --preset a100 --epochs 3 --task-limit 65

    # CPU smoke test (verifies imports, 3 tasks):
    python train/grpo_train.py --preset cpu --env-url http://localhost:8000
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.pr_review_env.client.pr_review_env_client import PRReviewEnvClient
from envs.pr_review_env.models import PRReviewAction, PRReviewObservation
from envs.pr_review_env.server.context_loader import load_review_config
from envs.pr_review_env.server.pr_review_env import PRReviewEnv
from envs.pr_review_env.server.tasks import load_tasks
from benchmarks.run_baselines import decide_final_verdict, heuristic_policy
from train.adaptive_router import prompt_route_context
from train.train_config import TrainingConfig  # noqa: E402 — after sys.path setup

# Heavy ML deps — guarded so file stays importable without trl installed
try:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import GRPOConfig, GRPOTrainer
    _ML_AVAILABLE = True
    _ML_IMPORT_ERROR: ImportError | None = None
except ImportError as exc:
    _ML_AVAILABLE = False
    _ML_IMPORT_ERROR = exc

# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a code reviewer. For each step output a JSON tool call only — no prose.
{"tool_name": "<name>", "arguments": {}}

Tools: check_security, check_quality, check_build_and_types, check_tests, check_config, submit_review, escalate.

For submit_review: {"tool_name": "submit_review", "arguments": {"verdict": "<approve|request_changes|reject>", "confidence": 0.9, "reasoning": "<one sentence>"}}
"""


def _review_config_lines(review_config: dict | None) -> list[str]:
    if not review_config:
        return []

    lines = ["", "Review configuration:"]
    architecture = review_config.get("architecture_summary")
    if architecture:
        lines.append(f"Architecture: {architecture}")
    critical_paths = review_config.get("critical_paths") or []
    if critical_paths:
        lines.append("Critical paths: " + ", ".join(critical_paths[:8]))
    domain_priorities = review_config.get("domain_priorities") or {}
    if domain_priorities:
        rendered = ", ".join(f"{key}={value}" for key, value in sorted(domain_priorities.items()))
        lines.append("Domain priorities: " + rendered)
    enabled_tools = review_config.get("enabled_tools") or []
    if enabled_tools:
        lines.append("Enabled external tools: " + ", ".join(enabled_tools[:12]))
    return lines


def build_obs_prompt(obs: PRReviewObservation, review_config: dict | None = None) -> str:
    parts = [
        f"Language: {obs.primary_language}",
        f"Files: {', '.join(obs.changed_file_types)}",
        f"Repo: {obs.repo_kind}",
        prompt_route_context(obs, review_config),
        "",
        f"PR: {obs.pr_description}",
        "",
        "Diff (truncated to 1500 chars):",
        obs.diff_str[:1500],
    ]
    parts += _review_config_lines(review_config)
    if obs.review_history:
        parts += ["", "Review so far:"] + obs.review_history[-4:]
    return "\n".join(parts)


def parse_action(text: str) -> PRReviewAction:
    """Extract last JSON object from model output. Falls back to submit_review on error."""
    payload = extract_action_payload(text)
    if payload is None:
        return PRReviewAction(
            tool_name="submit_review",
            arguments={"verdict": "approve", "confidence": 0.5, "reasoning": "parse error"},
        )

    tool = payload.get("tool_name", "submit_review")
    args = payload.get("arguments", {})
    if not isinstance(args, dict):
        args = {}
    return PRReviewAction(tool_name=tool, arguments=args)


def extract_action_payload(text: str) -> dict | None:
    """Return the last JSON object containing tool_name, or None."""
    decoder = json.JSONDecoder()
    payload = None
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and "tool_name" in candidate:
            payload = candidate
    return payload


ANALYSIS_TOOLS = {
    "check_security",
    "check_quality",
    "check_build_and_types",
    "check_tests",
    "check_config",
}


def _completion_to_text(completion) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content", completion))
    if isinstance(completion, list):
        for item in reversed(completion):
            if isinstance(item, dict) and "content" in item:
                return str(item["content"])
        return "\n".join(str(item) for item in completion)
    return str(completion)


def _action_with_state_args(action: PRReviewAction, obs: PRReviewObservation) -> PRReviewAction:
    args = dict(action.arguments)
    if action.tool_name in ANALYSIS_TOOLS:
        args.setdefault("diff_str", obs.diff_str)
    if action.tool_name == "submit_review":
        args.setdefault("reasoning", "Model submitted a final verdict.")
        args.setdefault("confidence", 0.5)
    if action.tool_name == "escalate":
        args.setdefault("reason", "Model requested human review.")
    return PRReviewAction(tool_name=action.tool_name, arguments=args)


def training_prompt(obs: PRReviewObservation, review_config: dict | None = None) -> str:
    return "\n\n".join(
        [
            SYSTEM_PROMPT.strip(),
            build_obs_prompt(obs, review_config),
            "Output exactly one JSON tool call.",
        ]
    )


def build_training_state_rows(
    cfg: TrainingConfig,
    review_config: dict | None = None,
) -> list[dict]:
    """Build trainable env states. Rewards are computed online, not precomputed."""
    tasks = load_tasks()
    if cfg.training_task_ids:
        tasks = [task for task in tasks if task.task_id in set(cfg.training_task_ids)]
    if cfg.training_task_limit and len(tasks) > cfg.training_task_limit:
        tasks = tasks[: cfg.training_task_limit]

    rows: list[dict] = []
    for task in tasks:
        env = PRReviewEnv(seed=7, review_config=review_config)
        obs = env.reset(task_id=task.task_id)
        replay_actions: list[dict] = []

        for teacher_action in heuristic_policy(obs):
            rows.append(
                {
                    "prompt": training_prompt(obs, review_config),
                    "task_id": task.task_id,
                    "replay_actions": list(replay_actions),
                }
            )
            obs = env.step(_action_with_state_args(teacher_action, obs))
            replay_actions.append(teacher_action.model_dump())
            if obs.done:
                break

        if not obs.done:
            rows.append(
                {
                    "prompt": training_prompt(obs, review_config),
                    "task_id": task.task_id,
                    "replay_actions": list(replay_actions),
                }
            )
            final_action = decide_final_verdict(obs)
            replay_actions.append(final_action.model_dump())

    return rows


def score_completion_locally(
    completion,
    task_id: str,
    replay_actions: list[dict] | None = None,
    review_config: dict | None = None,
) -> float:
    text = _completion_to_text(completion)
    if extract_action_payload(text) is None:
        return -0.75

    env = PRReviewEnv(seed=7, review_config=review_config)
    obs = env.reset(task_id=task_id)
    try:
        for payload in replay_actions or []:
            replay_action = PRReviewAction.model_validate(payload)
            obs = env.step(_action_with_state_args(replay_action, obs))
            if obs.done:
                return -0.50

        action = _action_with_state_args(parse_action(text), obs)
        obs = env.step(action)
    except Exception:
        return -0.75

    reward = float(obs.reward or 0.0)
    if obs.last_tool_result.get("error"):
        reward -= 0.75
    return round(reward, 3)


def make_env_reward_func(review_config: dict | None = None):
    """Create a TRL reward function that scores completions in PRReviewEnv."""

    def reward_func(completions, task_id=None, replay_actions=None, **kwargs):
        task_ids = task_id or kwargs.get("task_id")
        if task_ids is None:
            raise ValueError("GRPO reward function requires task_id dataset column")
        replays = replay_actions or kwargs.get("replay_actions") or [None] * len(completions)
        if isinstance(task_ids, str):
            task_ids = [task_ids] * len(completions)
        rewards = []
        for completion, tid, replay in zip(completions, task_ids, replays):
            rewards.append(score_completion_locally(completion, tid, replay, review_config))
        return rewards

    return reward_func


# ---------------------------------------------------------------------------
# Rollout — single episode
# ---------------------------------------------------------------------------

async def rollout_episode(
    env_client: PRReviewEnvClient,
    model,
    tokenizer,
    device,
    cfg: TrainingConfig,
    task_id: str | None = None,
    review_config: dict | None = None,
) -> dict:
    """Run one full episode. Returns {prompt, response, reward}."""
    reset_kwargs = {"task_id": task_id} if task_id else {}
    step_result = await env_client.reset(**reset_kwargs)
    obs = step_result.observation

    system_msg = {"role": "system", "content": SYSTEM_PROMPT}
    first_user = {"role": "user", "content": build_obs_prompt(obs, review_config)}
    messages = [system_msg, first_user]

    total_reward = 0.0
    last_response = ""

    for _ in range(cfg.max_steps_per_episode):
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )

        new_tokens = output_ids[0][input_ids.shape[-1]:]
        response_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        last_response = response_text

        action = _action_with_state_args(parse_action(response_text), obs)
        step_result = await env_client.step(action)
        obs = step_result.observation
        total_reward += step_result.reward or 0.0

        messages.append({"role": "assistant", "content": response_text})
        if obs.done:
            break
        messages.append({"role": "user", "content": build_obs_prompt(obs, review_config)})

    prompt_text = tokenizer.apply_chat_template(
        [system_msg, first_user], add_generation_prompt=True, tokenize=False
    )
    return {"prompt": prompt_text, "response": last_response, "reward": total_reward}


# ---------------------------------------------------------------------------
# Parallel trajectory collection
# ---------------------------------------------------------------------------

async def _collect_batch(
    env_url: str,
    model,
    tokenizer,
    device,
    cfg: TrainingConfig,
    task_ids: list[str],
    review_config: dict | None = None,
) -> list[dict]:
    """Collect one rollout per task_id, up to max_parallel_rollouts concurrent."""
    sem = asyncio.Semaphore(cfg.max_parallel_rollouts)

    async def _one(task_id: str) -> dict:
        async with sem:
            async with PRReviewEnvClient(base_url=env_url) as client:
                return await rollout_episode(client, model, tokenizer, device, cfg, task_id, review_config)

    return await asyncio.gather(*[_one(tid) for tid in task_ids])


async def collect_trajectories(
    env_url: str,
    model,
    tokenizer,
    device,
    cfg: TrainingConfig,
    review_config: dict | None = None,
) -> list[dict]:
    """Sample cfg.training_task_limit tasks, run cfg.num_generations rollouts each."""
    import random

    tasks = load_tasks()
    if cfg.training_task_ids:
        tasks = [t for t in tasks if t.task_id in set(cfg.training_task_ids)]
    if cfg.training_task_limit and len(tasks) > cfg.training_task_limit:
        tasks = random.sample(tasks, cfg.training_task_limit)

    # Expand: each task repeated num_generations times
    task_ids = [t.task_id for t in tasks for _ in range(cfg.num_generations)]

    t0 = time.time()
    trajectories = await _collect_batch(env_url, model, tokenizer, device, cfg, task_ids, review_config)
    elapsed = round(time.time() - t0, 1)
    rewards = [t["reward"] for t in trajectories]
    mean_r = sum(rewards) / max(len(rewards), 1)
    correct = sum(1 for t in trajectories if _verdict_correct(t))
    acc = correct / max(len(trajectories), 1)
    print(
        f"  Collected {len(trajectories)} trajectories in {elapsed}s — "
        f"mean_reward={mean_r:+.3f}  accuracy={acc:.3f}"
    )
    return trajectories


def _verdict_correct(traj: dict) -> bool:
    """Heuristic: response contains 'correct' verdict signal (can't check ground truth here)."""
    # The reward being > 0.8 is a reasonable proxy for a correct terminal verdict
    return traj["reward"] > 0.8


def build_dataset(rows: list[dict]) -> "Dataset":
    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# Training log callback
# ---------------------------------------------------------------------------

class RewardLogCallback:
    """Transformers TrainerCallback that appends one CSV row per logged step.

    Columns: step, loss, reward_mean, reward_std, kl
    Written to <output_dir>/training_log.csv — readable by generate_report.py.
    """

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("w", newline="") as f:
            csv.writer(f).writerow(["step", "loss", "reward_mean", "reward_std", "kl"])

    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001
        if not logs:
            return
        step = state.global_step
        row = [
            step,
            logs.get("loss", logs.get("train_loss", "")),
            logs.get("reward_mean", logs.get("rewards/mean", logs.get("reward", ""))),
            logs.get("reward_std", logs.get("rewards/std", "")),
            logs.get("kl", logs.get("kl_divergence", "")),
        ]
        with self.log_path.open("a", newline="") as f:
            csv.writer(f).writerow(row)


# ---------------------------------------------------------------------------
# Model loading with QLoRA
# ---------------------------------------------------------------------------

def load_model_qlora(cfg: TrainingConfig):
    """Load Qwen3-1.7B with 4-bit quantisation + LoRA adapters."""
    print(f"Loading {cfg.model_name} with QLoRA (4-bit={cfg.load_in_4bit}, r={cfg.lora.r})...")

    bnb_config = None
    if cfg.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if not cfg.load_in_4bit else None,
        device_map="auto",
        attn_implementation="eager",  # flash_attn optional; eager is safe everywhere
    )

    if cfg.load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_cfg = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.lora_alpha,
        target_modules=cfg.lora.target_modules,
        lora_dropout=cfg.lora.lora_dropout,
        bias=cfg.lora.bias,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # required for GRPO batch generation

    return model, tokenizer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO training for PR review (SLM-only, QLoRA)")
    parser.add_argument("--preset", default="t4",
                        choices=["t4", "a100", "v100", "h100", "cpu"],
                        help="Hardware preset (sets batch size, LoRA rank, parallelism)")
    parser.add_argument("--env-url", default=os.getenv("PR_REVIEW_ENV_URL", "http://localhost:8000"))
    parser.add_argument("--epochs", type=int, default=None, help="Override preset epoch count")
    parser.add_argument("--output-dir", default="./grpo_checkpoint")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-generations", type=int, default=None)
    parser.add_argument("--task-limit", type=int, default=None)
    parser.add_argument("--lora-r", type=int, default=None)
    parser.add_argument("--review-config", default="", help="Optional review_config.json generated from docs.")
    parser.add_argument("--report-to", default="none", choices=["none", "wandb", "tensorboard"])
    parser.add_argument("--wandb", action="store_true", help="Shorthand for --report-to wandb")
    args = parser.parse_args()

    if not _ML_AVAILABLE:
        raise RuntimeError(
            "ML deps not installed or incomplete.\n"
            f"Import error: {_ML_IMPORT_ERROR}\n"
            "Run: pip install trl peft bitsandbytes accelerate transformers datasets torch"
        )

    # Build config from preset then apply overrides
    preset_map = {"t4": TrainingConfig.for_t4, "a100": TrainingConfig.for_a100,
                  "v100": TrainingConfig.for_v100, "h100": TrainingConfig.for_h100,
                  "cpu": TrainingConfig.for_cpu}
    cfg = preset_map[args.preset]()
    cfg.output_dir = args.output_dir
    cfg.env_url = args.env_url
    if args.epochs is not None:
        cfg.num_train_epochs = args.epochs
    if args.lr is not None:
        cfg.learning_rate = args.lr
    if args.num_generations is not None:
        cfg.num_generations = args.num_generations
    if args.task_limit is not None:
        cfg.training_task_limit = args.task_limit
    if args.lora_r is not None:
        cfg.lora.r = args.lora_r
        cfg.lora.lora_alpha = args.lora_r * 2
    if args.wandb:
        cfg.report_to = "wandb"
    else:
        cfg.report_to = args.report_to
    review_config = load_review_config(args.review_config)

    est = cfg.estimate_epoch_time_minutes()
    print(f"Preset: {args.preset}  |  tasks={cfg.training_task_limit}  "
          f"generations={cfg.num_generations}  epochs={cfg.num_train_epochs}")
    print(f"Estimated time: ~{est} min/epoch × {cfg.num_train_epochs} = "
          f"~{round(est * cfg.num_train_epochs)} min total")
    if review_config:
        print(f"Loaded review config: {args.review_config}")

    # Load model
    model, tokenizer = load_model_qlora(cfg)
    device = next(p for p in model.parameters() if p.requires_grad).device

    print("\nBuilding online-reward training states...")
    rows = build_training_state_rows(cfg, review_config)
    dataset = build_dataset(rows)
    print(f"  Built {len(rows)} train states from {cfg.training_task_limit} tasks")

    # GRPO training config
    grpo_cfg = GRPOConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        num_generations=cfg.num_generations,
        generation_batch_size=cfg.num_generations,
        max_completion_length=cfg.max_new_tokens,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        report_to=cfg.report_to,
        # Memory optimisations
        gradient_checkpointing=True,
        bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        fp16=not torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )

    log_csv = Path(cfg.output_dir) / "training_log.csv"
    reward_cb = RewardLogCallback(log_csv)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[make_env_reward_func(review_config)],
        args=grpo_cfg,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[reward_cb],
    )

    print("\nStarting GRPO training...")
    trainer.train()
    trainer.save_model(cfg.output_dir)
    print(f"\nLoRA checkpoint saved  → {cfg.output_dir}")
    print(f"Training log (CSV)     → {log_csv}")
    print("To generate report:      python benchmarks/generate_report.py")


if __name__ == "__main__":
    main()
