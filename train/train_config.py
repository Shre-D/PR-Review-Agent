"""Hardware-aware training configuration for the PR Review GRPO agent.

Select a preset with:
    cfg = TrainingConfig.for_t4()      # Colab free / T4 16GB
    cfg = TrainingConfig.for_a100()    # Colab Pro+ / Kaggle / rented GPU
    cfg = TrainingConfig.for_cpu()     # smoke-test only, no real training

All presets use QLoRA so the 1.7B model fits in any modern GPU.
Full fine-tuning is intentionally NOT supported here — the goal is fast
iteration within a 2-4 hour GPU session.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LoRAConfig:
    r: int = 16
    lora_alpha: int = 32
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    lora_dropout: float = 0.05
    bias: str = "none"


@dataclass
class TrainingConfig:
    # Model
    model_name: str = "Qwen/Qwen3-1.7B"
    use_qlora: bool = True               # always True for this project
    load_in_4bit: bool = True
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    # GRPO hyperparameters
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8  # effective batch = 8
    learning_rate: float = 2e-4           # higher than full-FT because LoRA
    num_generations: int = 4             # rollouts per prompt (GRPO G)
    max_new_tokens: int = 128
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"

    # Rollout performance
    max_parallel_rollouts: int = 4        # async concurrent episodes
    max_steps_per_episode: int = 8

    # Task bank — use a curated fast subset during training
    # Full bank is used for evaluation only
    training_task_ids: list[str] = field(default_factory=list)  # empty = all tasks
    training_task_limit: int = 40         # cap tasks per epoch for speed

    # Environment
    env_url: str = "http://localhost:8000"
    env_backend: str = "heuristic"        # fastest; no external tool calls during training

    # Output
    output_dir: str = "./grpo_checkpoint"
    logging_steps: int = 1
    save_steps: int = 50
    eval_steps: int = 100

    # WandB / logging
    report_to: str = "none"              # set to "wandb" if token available

    # --- Hardware presets ---

    @classmethod
    def for_t4(cls) -> "TrainingConfig":
        """T4 16GB — Colab free tier, Kaggle.
        Fits QLoRA 1.7B easily. Conservative batch to avoid OOM.
        Expected: ~45 min/epoch on 40 tasks."""
        return cls(
            load_in_4bit=True,
            lora=LoRAConfig(r=16, lora_alpha=32),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            num_generations=4,
            max_parallel_rollouts=2,
            training_task_limit=40,
            learning_rate=2e-4,
        )

    @classmethod
    def for_a100(cls) -> "TrainingConfig":
        """A100 40GB — Colab Pro+, Kaggle P100, rented (RunPod/Lambda).
        Can push larger batch and more generations.
        Expected: ~20 min/epoch on 65 tasks."""
        return cls(
            load_in_4bit=True,          # still use 4-bit for speed headroom
            lora=LoRAConfig(r=32, lora_alpha=64),
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            num_generations=8,
            max_parallel_rollouts=8,
            training_task_limit=65,
            learning_rate=2e-4,
            num_train_epochs=3,
        )

    @classmethod
    def for_v100(cls) -> "TrainingConfig":
        """V100 16GB — similar to T4 but faster compute."""
        return cls(
            load_in_4bit=True,
            lora=LoRAConfig(r=16, lora_alpha=32),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            num_generations=4,
            max_parallel_rollouts=4,
            training_task_limit=50,
            learning_rate=2e-4,
        )

    @classmethod
    def for_h100(cls) -> "TrainingConfig":
        """H100 80GB — full bf16 (no quantization), large LoRA rank.
        Can fit the whole 65-task bank with 16 generations per prompt.
        Expected: ~10 min/epoch on 65 tasks."""
        return cls(
            load_in_4bit=False,          # H100 has 80GB — quantization is unnecessary
            lora=LoRAConfig(r=64, lora_alpha=128),
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,   # effective batch = 16
            num_train_epochs=5,
            num_generations=16,
            max_parallel_rollouts=16,
            training_task_limit=65,
            learning_rate=1e-4,
        )

    @classmethod
    def for_cpu(cls) -> "TrainingConfig":
        """CPU only — smoke test that the code runs. Not real training.
        Use 3 tasks, 2 generations, 1 epoch. GRPO requires at least two
        generations per prompt to compute advantages."""
        return cls(
            load_in_4bit=False,          # 4-bit requires CUDA
            lora=LoRAConfig(r=8, lora_alpha=16),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            num_train_epochs=1,
            num_generations=2,
            max_parallel_rollouts=1,
            training_task_limit=3,
            learning_rate=2e-4,
        )

    def estimate_epoch_time_minutes(self) -> float:
        """Rough estimate: rollout dominates, ~8s per episode on T4."""
        episodes = self.training_task_limit * self.num_generations
        batches = episodes / max(1, self.max_parallel_rollouts)
        seconds_per_batch = 8  # heuristic: ~2s/step * 4 steps avg
        return round(batches * seconds_per_batch / 60, 1)
