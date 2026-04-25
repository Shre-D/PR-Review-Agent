# RL Training Loop: GRPO, LoRA, and Beyond

The training loop fine-tunes `Qwen/Qwen3-1.7B` to output JSON tool calls for the
PR review environment.

The implementation lives in `train/grpo_train.py`.

## Why GRPO

GRPO, Group Relative Policy Optimization, is useful when we can score generated
outputs but do not have a single perfect target completion for every prompt.

For each prompt, the trainer samples multiple completions. The environment
reward function scores each completion. GRPO then updates the policy toward
actions that do better than other completions in the same group.

This is a good match for tool routing:

- there may be more than one acceptable first tool
- shorter routes are better than redundant routes
- the final verdict must match evidence, not just syntax
- invalid JSON should be discouraged immediately

## Prompt Construction

Each training row contains a prompt built from the current environment state.

The system prompt requires one JSON tool call and lists the available tools. The
user portion includes:

- language
- changed file types
- repository kind
- adaptive route context
- PR description
- truncated diff
- short loader context
- recent review history, if any

The prompt ends with:

```text
Output exactly one JSON tool call.
```

## Replayable Training States

Training does not roll out full async episodes through a remote server. Instead,
the script builds replayable states from the heuristic policy.

For each task:

1. reset `PRReviewEnv`
2. create a prompt for the current observation
3. store prior actions needed to replay that state
4. step through heuristic actions
5. repeat until the episode reaches a terminal state

Each row includes:

```json
{
  "prompt": "...",
  "task_id": "py_sql_injection",
  "replay_actions": [],
  "review_config": {"extraction_method": "task_short_structural"}
}
```

At reward time, the generated completion is parsed, the environment is reset,
the replay actions are applied, and the generated action is scored in the live
environment.

This keeps training deterministic, local, and easier to debug.

## LoRA and QLoRA

The base model is not fully fine-tuned. The training script uses PEFT LoRA
adapters. On memory-constrained GPUs it uses QLoRA-style 4-bit loading through
bitsandbytes.

The presets are:

| Preset | Use Case | Notes |
|---|---|---|
| `t4` | smoke/low-cost GPU | 4-bit, smaller batch, fewer generations |
| `a100` | Hugging Face Jobs | 4-bit, larger LoRA rank, full 78-task bank |
| `h100` | HPC | bf16, larger LoRA rank, more generations |
| `cpu` | import/data smoke only | not real training |

The current production command shape is:

```bash
PR_REVIEW_TOOL_BACKEND=heuristic \
python train/grpo_train.py \
  --preset a100 \
  --task-bank all \
  --task-loader-mode short \
  --task-limit 78 \
  --epochs 3 \
  --num-generations 8 \
  --output-dir ./grpo_checkpoint
```

## Reward Function During Training

The GRPO trainer receives `make_env_reward_func(...)`.

The reward function:

1. extracts the generated JSON action
2. returns `RAW_FLOOR` (`−1.5`) for malformed output
3. resets the environment to the task
4. replays prior state actions
5. applies the generated action
6. returns the **raw** environment reward

Raw rewards go directly to GRPO — no normalization. This is critical. An
earlier version normalized rewards into `[0.01, 0.99]`, which compressed the
gap between a relevant tool (`0.3`) and an irrelevant tool (`0.1`) down to
~0.018. GRPO could not compute meaningful advantages from that, and reward
variance collapsed to zero.

The current raw scale has clear gaps:

| Outcome | Raw Reward |
|---|---|
| Evidence-backed correct submit | `+1.0` to `+1.4` |
| Novel relevant tool | `+0.3` |
| Novel irrelevant tool | `+0.1` |
| Early correct submit | `+0.2` |
| Duplicate tool | `−0.4` |
| Wrong verdict | `−0.8` to `−1.0` |
| Malformed/error | `−1.5` |

This means the model is trained against the same reward code used by baseline
and trained-model evaluation. The ordering is explicit and the gaps are wide
enough for GRPO to optimize tool-use behavior instead of verdict guessing.

## HF Jobs Path

The HF Jobs launcher is `scripts/run_hf_job.sh`.

It calls the UV script `scripts/hf_train_job.py`, which:

- clones the repository from `REPO_URL`
- installs `.[server,train,dev]`
- rebuilds the combined task bank
- runs GRPO training
- evaluates baselines and the trained checkpoint
- pushes the LoRA adapter to `HF_HUB_MODEL_ID`
- uploads reward artifacts and the training log

Required environment:

```bash
hf auth login
export REPO_URL="https://github.com/<user>/<repo>.git"
export HF_HUB_MODEL_ID="<user>/pr-review-qwen3-1p7b"
bash scripts/run_hf_job.sh
```

## HPC Path

The HPC path is:

```bash
bash hpc/setup_env.sh
sbatch hpc/train_h100.slurm
```

The setup script creates a Python 3.11 environment, installs training
dependencies, and pre-downloads `Qwen/Qwen3-1.7B`. The Slurm job trains with the
H100 preset, runs baseline evaluation, runs trained-model evaluation, and writes
artifacts to scratch.

## Beyond the First Run

The first trained checkpoint should answer whether the learned policy can beat
the heuristic baseline on the 78-task benchmark. After that, the useful next
experiments are:

- SFT warm-start from `train/export_teacher_traces.py`
- train/validation split rather than using all tasks for tuning
- compare `short` vs `empty` loader mode
- increase comprehensive multi-file tasks
- add task variants that test clean approvals, not only vulnerabilities
- evaluate hybrid tool backend separately from heuristic backend

The key constraint remains: the deployed router should stay small and structured.
The value is in evidence routing, not in hiding a large reviewer behind the
benchmark.
