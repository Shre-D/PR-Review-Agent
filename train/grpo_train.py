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
    python train/grpo_train.py --preset a100 --epochs 3 --task-bank all --task-limit 78

    # CPU smoke test (verifies imports, 3 tasks):
    python train/grpo_train.py --preset cpu --env-url http://localhost:8000
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from threading import Lock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.pr_review_env.models import PRReviewAction, PRReviewObservation
from envs.pr_review_env.server.context_loader import load_review_config
from envs.pr_review_env.server.grader import reward_floor, reward_near_floor
from envs.pr_review_env.server.pr_review_env import PRReviewEnv
from envs.pr_review_env.server.tasks import load_tasks, task_review_config
from benchmarks.run_baselines import decide_final_verdict, heuristic_policy
from train.adaptive_router import prompt_route_context, review_requirements
from train.train_config import TrainingConfig  # noqa: E402 — after sys.path setup

# Heavy ML deps — guarded so file stays importable without trl installed
try:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainerCallback,
    )
    from trl import GRPOConfig, GRPOTrainer
    _ML_AVAILABLE = True
    _ML_IMPORT_ERROR: ImportError | None = None
except ImportError as exc:
    _ML_AVAILABLE = False
    _ML_IMPORT_ERROR = exc

    class TrainerCallback:  # type: ignore[no-redef]
        """Fallback when transformers isn't installed (smoke-import only)."""

# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a code reviewer. For each step output a JSON tool call only — no prose.
Do not output hidden reasoning, <think> blocks, markdown, or explanations.
{"tool_name": "<name>", "arguments": {}}

Tools: check_security, check_quality, check_build_and_types, check_tests, check_config, submit_review, escalate.

For submit_review: {"tool_name": "submit_review", "arguments": {"verdict": "<approve|request_changes|reject>", "confidence": <0.0-1.0>, "reasoning": "<one sentence>"}}
Calibrate confidence: high (>=0.8) only when at least min_tools have run and findings are unambiguous.
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
    route = review_requirements(obs, review_config)
    return "\n\n".join(
        [
            SYSTEM_PROMPT.strip(),
            build_obs_prompt(obs, review_config),
            (
                "Routing rule: collect at least "
                f"{route['min_tools']} evidence tools before any terminal verdict."
            ),
            (
                "Prefer an evidence tool first. Early final verdicts are redirected "
                "for a short evidence-gathering window before they can end the review."
            ),
            "Output exactly one JSON tool call.",
        ]
    )


def _read_holdout_ids(path: str | None) -> set[str]:
    if not path:
        return set()
    holdout_path = Path(path)
    if not holdout_path.exists():
        return set()
    return {
        line.strip()
        for line in holdout_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def build_training_state_rows(
    cfg: TrainingConfig,
    review_config: dict | None = None,
    holdout_ids: set[str] | None = None,
) -> list[dict]:
    """Build trainable env states. Rewards are computed online, not precomputed."""
    tasks = load_tasks(cfg.tasks_file)
    if cfg.training_task_ids:
        tasks = [task for task in tasks if task.task_id in set(cfg.training_task_ids)]
    if holdout_ids:
        tasks = [task for task in tasks if task.task_id not in holdout_ids]
    if cfg.training_task_limit and len(tasks) > cfg.training_task_limit:
        tasks = tasks[: cfg.training_task_limit]

    rows: list[dict] = []
    for task in tasks:
        task_config = review_config or task_review_config(task, mode=cfg.task_loader_mode)
        env = PRReviewEnv(seed=7, task_path=cfg.tasks_file, review_config=task_config)
        obs = env.reset(task_id=task.task_id)
        replay_actions: list[dict] = []

        for teacher_action in heuristic_policy(obs):
            rows.append(
                {
                    "prompt": training_prompt(obs, task_config),
                    "task_id": task.task_id,
                    "replay_actions": list(replay_actions),
                    "review_config": task_config,
                    "route": review_requirements(obs, task_config),
                }
            )
            obs = env.step(_action_with_state_args(teacher_action, obs))
            replay_actions.append(teacher_action.model_dump())
            if obs.done:
                break

        if not obs.done:
            rows.append(
                {
                    "prompt": training_prompt(obs, task_config),
                    "task_id": task.task_id,
                    "replay_actions": list(replay_actions),
                    "review_config": task_config,
                    "route": review_requirements(obs, task_config),
                }
            )
            final_action = decide_final_verdict(obs)
            replay_actions.append(final_action.model_dump())

    return rows


_ALL_TOOLS = (
    "check_security",
    "check_quality",
    "check_build_and_types",
    "check_tests",
    "check_config",
    "submit_review",
    "escalate",
    "invalid",
)


class TrainingMetricsCounter:
    """Thread-safe per-batch aggregator for env-side metrics.

    `score_completion_locally` writes one record per completion. The
    `RewardLogCallback` reads + clears it on every `on_log`, producing one
    CSV/W&B row per logged step.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._reset()

    def _reset(self) -> None:
        self.tool_counts: dict[str, int] = defaultdict(int)
        self.raw_rewards: list[float] = []
        self.normalized_rewards: list[float] = []
        self.terminal_count: int = 0
        self.terminal_correct: int = 0
        self.parse_failures: int = 0
        self.env_errors: int = 0

    def record(
        self,
        tool_name: str,
        normalized_reward: float,
        raw_reward: float | None = None,
        terminal: bool = False,
        correct: bool | None = None,
        parse_failed: bool = False,
        env_error: bool = False,
    ) -> None:
        with self._lock:
            self.tool_counts[tool_name] += 1
            self.normalized_rewards.append(normalized_reward)
            if raw_reward is not None:
                self.raw_rewards.append(raw_reward)
            if terminal:
                self.terminal_count += 1
                if correct:
                    self.terminal_correct += 1
            if parse_failed:
                self.parse_failures += 1
            if env_error:
                self.env_errors += 1

    def flush(self) -> dict[str, float]:
        with self._lock:
            total = sum(self.tool_counts.values()) or 1
            snapshot: dict[str, float] = {
                "n_completions": float(total),
                "raw_reward_mean": (
                    sum(self.raw_rewards) / len(self.raw_rewards)
                    if self.raw_rewards
                    else 0.0
                ),
                "normalized_reward_mean": (
                    sum(self.normalized_rewards) / len(self.normalized_rewards)
                    if self.normalized_rewards
                    else 0.0
                ),
                "parse_failure_rate": self.parse_failures / total,
                "env_error_rate": self.env_errors / total,
                "terminal_rate": self.terminal_count / total,
                "terminal_accuracy": (
                    self.terminal_correct / self.terminal_count
                    if self.terminal_count
                    else 0.0
                ),
            }
            for tool in _ALL_TOOLS:
                snapshot[f"tool_frac/{tool}"] = self.tool_counts.get(tool, 0) / total
                snapshot[f"tool_count/{tool}"] = float(self.tool_counts.get(tool, 0))
            self._reset()
            return snapshot


def score_completion_locally(
    completion,
    task_id: str,
    replay_actions: list[dict] | None = None,
    review_config: dict | None = None,
    task_path: str | None = None,
    counter: TrainingMetricsCounter | None = None,
) -> float:
    text = _completion_to_text(completion)
    payload = extract_action_payload(text)
    if payload is None:
        floor = reward_floor()
        if counter is not None:
            counter.record(
                tool_name="invalid",
                normalized_reward=floor,
                raw_reward=-2.83,
                parse_failed=True,
            )
        return floor

    env = PRReviewEnv(seed=7, task_path=task_path, review_config=review_config)
    obs = env.reset(task_id=task_id)
    expected_verdict = obs.metadata.get("expected_verdict") if isinstance(obs.metadata, dict) else None
    try:
        for replay_payload in replay_actions or []:
            replay_action = PRReviewAction.model_validate(replay_payload)
            obs = env.step(_action_with_state_args(replay_action, obs))
            if obs.done:
                floor = reward_floor()
                if counter is not None:
                    counter.record(
                        tool_name=str(payload.get("tool_name", "invalid")),
                        normalized_reward=floor,
                        raw_reward=-2.83,
                        env_error=True,
                    )
                return floor

        parsed = parse_action(text)
        action = _action_with_state_args(parsed, obs)
        obs = env.step(action)
    except Exception:
        floor = reward_floor()
        if counter is not None:
            counter.record(
                tool_name=str(payload.get("tool_name", "invalid")),
                normalized_reward=floor,
                raw_reward=-2.83,
                env_error=True,
            )
        return floor

    normalized = float(obs.reward or 0.0)
    if obs.last_tool_result.get("error"):
        normalized = reward_near_floor()
        if counter is not None:
            counter.record(
                tool_name=action.tool_name,
                normalized_reward=normalized,
                raw_reward=-2.55,
                env_error=True,
            )
        return round(normalized, 3)

    if counter is not None:
        terminal = action.tool_name in {"submit_review", "escalate"}
        correct: bool | None = None
        if terminal and expected_verdict is not None:
            submitted = obs.last_tool_result.get("verdict", action.tool_name)
            correct = str(submitted).strip().lower() == str(expected_verdict).strip().lower()
        # Approximate raw reward by inverting the normalize_reward map.
        # normalize_reward: norm = 0.01 + (raw - RAW_MIN) / (RAW_MAX - RAW_MIN) * 0.98
        # Solve for raw:
        raw_min, raw_max = -2.83, 1.51
        raw = (normalized - 0.01) / 0.98 * (raw_max - raw_min) + raw_min
        counter.record(
            tool_name=action.tool_name,
            normalized_reward=normalized,
            raw_reward=raw,
            terminal=terminal,
            correct=correct,
        )
    return round(normalized, 3)


def _clean_review_config_payload(review_config: dict | None) -> dict | None:
    """Drop Dataset-introduced nulls from sparse nested ReviewConfig dicts."""
    if not isinstance(review_config, dict):
        return review_config

    cleaned = dict(review_config)
    for key in ("tool_weights", "domain_priorities", "author_depth"):
        value = cleaned.get(key)
        if isinstance(value, dict):
            cleaned[key] = {k: v for k, v in value.items() if v is not None}
    for key in ("enabled_tools", "planned_tools", "critical_paths", "custom_rules"):
        value = cleaned.get(key)
        if isinstance(value, list):
            cleaned[key] = [item for item in value if item is not None]
    return cleaned


def make_env_reward_func(
    review_config: dict | None = None,
    task_path: str | None = None,
    counter: TrainingMetricsCounter | None = None,
):
    """Create a TRL reward function that scores completions in PRReviewEnv.

    If `counter` is provided, every completion is recorded with its tool
    choice, normalized reward, approximate raw reward, and terminal-correctness
    flag. The `RewardLogCallback` flushes the counter at every logged step.
    """

    def reward_func(completions, task_id=None, replay_actions=None, **kwargs):
        task_ids = task_id or kwargs.get("task_id")
        if task_ids is None:
            raise ValueError("GRPO reward function requires task_id dataset column")
        replays = replay_actions or kwargs.get("replay_actions") or [None] * len(completions)
        review_configs = kwargs.get("review_config") or [review_config] * len(completions)
        if isinstance(task_ids, str):
            task_ids = [task_ids] * len(completions)
        if isinstance(review_configs, dict) or review_configs is None:
            review_configs = [review_configs] * len(completions)
        rewards = []
        for completion, tid, replay, cfg_payload in zip(completions, task_ids, replays, review_configs):
            rewards.append(
                score_completion_locally(
                    completion,
                    tid,
                    replay,
                    _clean_review_config_payload(cfg_payload),
                    task_path,
                    counter=counter,
                )
            )
        return rewards

    return reward_func


def build_dataset(rows: list[dict]) -> "Dataset":
    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# Training log callback
# ---------------------------------------------------------------------------

_BASE_COLUMNS = ["step", "loss", "reward_mean", "reward_std", "kl"]
_ENV_COLUMNS = [
    "raw_reward_mean",
    "normalized_reward_mean",
    "terminal_rate",
    "terminal_accuracy",
    "parse_failure_rate",
    "env_error_rate",
    "n_completions",
]
_TOOL_FRAC_COLUMNS = [f"tool_frac/{tool}" for tool in _ALL_TOOLS]
_TOOL_COUNT_COLUMNS = [f"tool_count/{tool}" for tool in _ALL_TOOLS]
LOG_COLUMNS = _BASE_COLUMNS + _ENV_COLUMNS + _TOOL_FRAC_COLUMNS + _TOOL_COUNT_COLUMNS


class RewardLogCallback(TrainerCallback):
    """TrainerCallback that appends one CSV row per logged step and mirrors
    custom env-side metrics to W&B when reporting is enabled.

    CSV columns:
        step, loss, reward_mean, reward_std, kl,
        raw_reward_mean, normalized_reward_mean,
        terminal_rate, terminal_accuracy,
        parse_failure_rate, env_error_rate, n_completions,
        tool_frac/<tool>...   (fraction of completions that picked this tool)
        tool_count/<tool>...  (absolute count this logging window)

    Written to <output_dir>/training_log.csv — readable by generate_report.py.
    """

    def __init__(
        self,
        log_path: Path,
        counter: TrainingMetricsCounter | None = None,
        report_to: str = "none",
    ) -> None:
        super().__init__()
        self.log_path = log_path
        self.counter = counter
        self.report_to = report_to
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("w", newline="") as f:
            csv.writer(f).writerow(LOG_COLUMNS)

    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001
        if not logs:
            return
        step = state.global_step
        env_metrics = self.counter.flush() if self.counter is not None else {}

        row_dict: dict[str, object] = {
            "step": step,
            "loss": logs.get("loss", logs.get("train_loss", "")),
            "reward_mean": logs.get("reward_mean", logs.get("rewards/mean", logs.get("reward", ""))),
            "reward_std": logs.get("reward_std", logs.get("rewards/std", "")),
            "kl": logs.get("kl", logs.get("kl_divergence", "")),
        }
        for col in _ENV_COLUMNS + _TOOL_FRAC_COLUMNS + _TOOL_COUNT_COLUMNS:
            row_dict[col] = env_metrics.get(col, "")

        with self.log_path.open("a", newline="") as f:
            csv.writer(f).writerow([row_dict.get(col, "") for col in LOG_COLUMNS])

        if self.report_to == "wandb" and env_metrics:
            try:
                import wandb

                if wandb.run is not None:
                    wandb_payload = {
                        f"env/{key.split('/')[-1]}" if "/" not in key else f"env/{key}": value
                        for key, value in env_metrics.items()
                        if not key.startswith("tool_")
                    }
                    for tool in _ALL_TOOLS:
                        wandb_payload[f"env/tool_frac/{tool}"] = env_metrics.get(f"tool_frac/{tool}", 0.0)
                        wandb_payload[f"env/tool_count/{tool}"] = env_metrics.get(f"tool_count/{tool}", 0.0)
                    wandb.log(wandb_payload, step=step)
            except ImportError:
                pass


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
                        help="Hardware preset (sets batch size, LoRA rank, generations, and task limit)")
    parser.add_argument("--env-url", default=os.getenv("PR_REVIEW_ENV_URL", "http://localhost:8000"))
    parser.add_argument("--epochs", type=int, default=None, help="Override preset epoch count")
    parser.add_argument("--output-dir", default="./grpo_checkpoint")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-generations", type=int, default=None)
    parser.add_argument("--task-limit", type=int, default=None)
    parser.add_argument("--tasks-file", "--task-bank", dest="tasks_file", default="all")
    parser.add_argument(
        "--task-loader-mode",
        default="short",
        choices=["short", "empty", "full", "off", "none"],
        help="Per-task loader config included in prompts/rewards.",
    )
    parser.add_argument("--lora-r", type=int, default=None)
    parser.add_argument("--review-config", default="", help="Optional review_config.json generated from docs.")
    parser.add_argument(
        "--holdout-file",
        default=str(ROOT / "tasks" / "holdout_ids.txt"),
        help="File listing task_ids excluded from training. Defaults to tasks/holdout_ids.txt; pass empty string to disable.",
    )
    parser.add_argument("--report-to", default="none", choices=["none", "wandb", "tensorboard"])
    parser.add_argument("--wandb", action="store_true", help="Shorthand for --report-to wandb")
    parser.add_argument(
        "--hub-model-id",
        default=os.getenv("HF_HUB_MODEL_ID", ""),
        help="Optional Hub repo ID for pushing the trained LoRA adapter, e.g. user/pr-review-qwen3-1p7b.",
    )
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
    cfg.tasks_file = args.tasks_file
    cfg.task_loader_mode = args.task_loader_mode
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

    # Propagate env_backend to the env var read by tools.py.
    # TrainingConfig defaults to "heuristic" — no subprocess tool calls during training.
    # Must be set before any PRReviewEnv is instantiated (including inside reward_func).
    os.environ.setdefault("PR_REVIEW_TOOL_BACKEND", cfg.env_backend)

    est = cfg.estimate_epoch_time_minutes()
    print(f"Preset: {args.preset}  |  tasks={cfg.training_task_limit}  "
          f"generations={cfg.num_generations}  epochs={cfg.num_train_epochs}")
    print(f"Task bank: {cfg.tasks_file}  |  task loader: {cfg.task_loader_mode}")
    print(f"Estimated time: ~{est} min/epoch × {cfg.num_train_epochs} = "
          f"~{round(est * cfg.num_train_epochs)} min total")
    if review_config:
        print(f"Loaded review config: {args.review_config}")

    # Load model
    model, tokenizer = load_model_qlora(cfg)

    holdout_ids = _read_holdout_ids(args.holdout_file)
    if holdout_ids:
        print(f"Holdout: excluding {len(holdout_ids)} task_ids from training (file={args.holdout_file})")

    print("\nBuilding online-reward training states...")
    rows = build_training_state_rows(cfg, review_config, holdout_ids=holdout_ids)
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
        push_to_hub=bool(args.hub_model_id),
        hub_model_id=args.hub_model_id or None,
    )

    log_csv = Path(cfg.output_dir) / "training_log.csv"
    metrics_counter = TrainingMetricsCounter()
    reward_cb = RewardLogCallback(log_csv, counter=metrics_counter, report_to=cfg.report_to)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[make_env_reward_func(review_config, cfg.tasks_file, counter=metrics_counter)],
        args=grpo_cfg,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[reward_cb],
    )

    print("\nStarting GRPO training...")
    trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    if args.hub_model_id:
        trainer.push_to_hub()
    print(f"\nLoRA checkpoint saved  → {cfg.output_dir}")
    print(f"Training log (CSV)     → {log_csv}")
    if args.hub_model_id:
        print(f"Hub checkpoint         → {args.hub_model_id}")
    print("To generate report:      python benchmarks/generate_report.py")


if __name__ == "__main__":
    main()
