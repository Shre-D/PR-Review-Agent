# Hugging Face and HPC Commands

## One-time Hugging Face login

```bash
hf auth login
```

## Training on Hugging Face Jobs

### Full run

```bash
hf jobs uv run \
  --flavor a100-large \
  --timeout 3h \
  --secrets HF_TOKEN \
  --env REPO_URL=https://github.com/Shre-D/PR-Review-Agent.git \
  --env REPO_REF=development \
  --env HF_HUB_MODEL_ID=shred12/pr-review-qwen3-1p7b-patience \
  --env PRESET=a100 \
  --env EPOCHS=3 \
  --env TASK_LIMIT=78 \
  --env NUM_GENERATIONS=8 \
  --env EVAL_LIMIT=0 \
  --env TASK_LOADER_MODE=short \
  --detach \
  scripts/hf_train_job.py
```

### Small validation run

```bash
hf jobs uv run \
  --flavor t4-small \
  --timeout 2h \
  --secrets HF_TOKEN \
  --env REPO_URL=https://github.com/Shre-D/PR-Review-Agent.git \
  --env REPO_REF=development \
  --env HF_HUB_MODEL_ID=shred12/pr-review-qwen3-1p7b-patience-smoke \
  --env PRESET=t4 \
  --env EPOCHS=1 \
  --env TASK_LIMIT=20 \
  --env NUM_GENERATIONS=4 \
  --env EVAL_LIMIT=10 \
  --env TASK_LOADER_MODE=short \
  --detach \
  scripts/hf_train_job.py
```

## Evaluating a trained checkpoint on Hugging Face Jobs

### Smoke eval

```bash
hf jobs uv run \
  --flavor l4x1 \
  --timeout 1h \
  --secrets HF_TOKEN \
  --env REPO_URL=https://github.com/Shre-D/PR-Review-Agent.git \
  --env REPO_REF=development \
  --env HF_HUB_MODEL_ID=shred12/pr-review-qwen3-1p7b-patience-smoke \
  --env PRESET=t4 \
  --env EVAL_LIMIT=10 \
  --env TASK_LOADER_MODE=short \
  --detach \
  scripts/hf_eval_job.py
```

### Full eval

```bash
hf jobs uv run \
  --flavor l4x1 \
  --timeout 2h \
  --secrets HF_TOKEN \
  --env REPO_URL=https://github.com/Shre-D/PR-Review-Agent.git \
  --env REPO_REF=development \
  --env HF_HUB_MODEL_ID=shred12/pr-review-qwen3-1p7b-patience-smoke \
  --env PRESET=t4 \
  --env EVAL_LIMIT=0 \
  --env TASK_LOADER_MODE=short \
  --detach \
  scripts/hf_eval_job.py
```

## Hugging Face job inspection

```bash
hf jobs inspect <job_id>
hf jobs logs <job_id> --tail 200
hf jobs cancel <job_id>
hf jobs ps
```

## Hugging Face model repo checks

```bash
hf models info shred12/pr-review-qwen3-1p7b-patience-smoke
hf models info shred12/pr-review-qwen3-1p7b-patience
hf download shred12/pr-review-qwen3-1p7b-patience-smoke --local-dir /tmp/pr-review-checkpoint
```

## HPC sync to `/scratch/kulkarnis`

```bash
mkdir -p /scratch/kulkarnis/PR-Review-Agent

rsync -av --delete \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  --exclude '.mypy_cache/' \
  --exclude '.benchmarks/' \
  --exclude 'logs/' \
  --exclude 'grpo_checkpoint/' \
  --exclude 'rewards/' \
  --exclude 'hf_cache/' \
  --exclude '*.pyc' \
  ./ \
  kulkarnis@hpc.bits-hyderabad.ac.in:/scratch/kulkarnis/PR-Review-Agent/
```

## HPC setup and run

```bash
ssh kulkarnis@hpc.bits-hyderabad.ac.in
cd /scratch/kulkarnis/PR-Review-Agent
bash hpc/setup_env.sh
```

```bash
cd /scratch/kulkarnis/PR-Review-Agent
REPO_DIR=/scratch/kulkarnis/PR-Review-Agent \
ENV_DIR=/scratch/kulkarnis/envs/pr-review \
HF_CACHE=/scratch/kulkarnis/hf_cache \
OUTPUT_DIR=/scratch/kulkarnis/pr-review/grpo_checkpoint \
REWARDS_DIR=/scratch/kulkarnis/pr-review/rewards \
sbatch hpc/train_h100.slurm
```

## HPC job monitoring

```bash
squeue -u "$USER"
tail -f logs/grpo_<jobid>.out
tail -f logs/grpo_<jobid>.err
```
• Ran hf jobs uv run --flavor t4-small --timeout 2h --secrets HF_TOKEN --env REPO_URL=https://github.com/Shre-D/PR-Review-Agent.git
  │ --env REPO_REF=development --env HF_HUB_MODEL_ID=shred12/pr-review-qwen3-1p7b-patience-smoke --env PRESET=t4 --env EPOCHS=1
  │ --env TASK_LIMIT=20 --env NUM_GENERATIONS=4 --env EVAL_LIMIT=10 --env TASK_LOADER_MODE=short --detach scripts/hf_train_job.py