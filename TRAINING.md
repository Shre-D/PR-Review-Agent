# Training Guide: PR Review GRPO Agent

Training the Qwen3-1.7B router via GRPO on rented or free GPU hardware.

**Design constraint:** No LLM calls during training. The RL router is always a SLM.
The context loader (org docs → config) is an offline step run once before training starts.

**Current status:** `train/grpo_train.py` uses TRL GRPO with an online environment
reward function. Each generated tool call is parsed, replayed into `PRReviewEnv`,
and scored from the live environment reward.

---

## Hardware Quick Reference

| Platform | GPU | VRAM | Est. Time / Epoch | Cost |
|----------|-----|------|-------------------|------|
| Colab Free | T4 | 15 GB | ~45 min (40 tasks) | Free |
| Colab Pro+ | A100 | 40 GB | ~20 min (65 tasks) | ~$10/hr |
| Kaggle | T4×2 | 15 GB | ~40 min | Free (30hr/wk) |
| RunPod | A100 | 40/80 GB | ~20 min | ~$1.50/hr |
| Lambda Labs | A100 | 40 GB | ~20 min | ~$1.10/hr |
| HuggingFace ZeroGPU | A10G | 24 GB | ~30 min | Free (limited) |

**Minimum:** T4 16GB. Anything less requires reducing `num_generations` to 2 and `training_task_limit` to 20.

---

## 1. Environment Setup

### Colab / Kaggle (run in a cell)

```bash
# 1. Clone the repo
!git clone https://github.com/YOUR_USERNAME/pr-review-agent.git
%cd pr-review-agent

# 2. Install training deps
!pip install -q \
  "trl>=0.9.0" \
  "transformers>=4.45.0" \
  "peft>=0.12.0" \
  "bitsandbytes>=0.43.0" \
  "accelerate>=0.30.0" \
  "datasets>=2.20.0" \
  "torch>=2.3.0" \
  "fastapi" "uvicorn" "httpx" "pyyaml" "fastmcp"

# 3. Verify GPU
!python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory // 1e9, 'GB')"
```

### RunPod / Lambda Labs (SSH)

```bash
git clone https://github.com/YOUR_USERNAME/pr-review-agent.git
cd pr-review-agent

pip install trl>=0.9.0 transformers>=4.45.0 peft>=0.12.0 \
  bitsandbytes>=0.43.0 accelerate>=0.30.0 datasets>=2.20.0 \
  fastapi uvicorn httpx pyyaml fastmcp

# Optional: flash attention for 20-30% speedup on A100
pip install flash-attn --no-build-isolation
```

---

## 2. Start the Environment Server

The env server must be running before training starts. Start it as a background process.

### Option A: Background process (Colab / SSH)

```bash
# In a terminal or notebook cell:
PR_REVIEW_TOOL_BACKEND=heuristic \
  uvicorn envs.pr_review_env.server.app:app \
  --host 0.0.0.0 --port 8000 &

# Verify it's up
sleep 3 && curl http://localhost:8000/health
```

### Option B: Colab notebook cell (use subprocess)

```python
import subprocess, time

server = subprocess.Popen(
    ["uvicorn", "envs.pr_review_env.server.app:app",
     "--host", "0.0.0.0", "--port", "8000"],
    env={**__import__("os").environ, "PR_REVIEW_TOOL_BACKEND": "heuristic"},
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)
time.sleep(4)

import urllib.request
urllib.request.urlopen("http://localhost:8000/health")
print("Server ready")
```

### Option C: Use a deployed Space (if local GPU is separate from env server)

```python
ENV_URL = "https://YOUR_USERNAME-pr-review-env.hf.space"
# Set this in the training command below
```

---

## 3. (Optional) Generate Org Config

If you want the model to train with org-specific context (architecture summary, custom rules), run the context loader first. **This is offline-only — not called during training.**

```bash
# Structural parsing only (no model, deterministic)
python -m envs.pr_review_env.server.context_loader \
  --docs-dir docs/ \
  --output review_config.json

# Same structural loader, preserving workflow compatibility with rule-extraction jobs
python -m envs.pr_review_env.server.context_loader \
  --docs-dir docs/ \
  --output review_config.json \
  --extract-rules
```

If you skip this step, training uses default reward weights and no custom rules. The
current checked-in loader is deterministic and structural; model-assisted rule
extraction is reserved for the post-training loader benchmark.

---

## 4. Run Training

### T4 (Colab Free / Kaggle)

```bash
PR_REVIEW_TOOL_BACKEND=heuristic \
.venv/bin/python train/grpo_train.py \
  --preset t4 \
  --output-dir ./grpo_checkpoint \
  --epochs 2 \
  --task-limit 40 \
  --num-generations 4
```

### A100 (Colab Pro+ / RunPod / Lambda)

```bash
PR_REVIEW_TOOL_BACKEND=heuristic \
.venv/bin/python train/grpo_train.py \
  --preset a100 \
  --output-dir ./grpo_checkpoint \
  --epochs 3
```

### Custom config

```bash
PR_REVIEW_TOOL_BACKEND=heuristic \
.venv/bin/python train/grpo_train.py \
  --preset t4 \
  --output-dir ./grpo_checkpoint \
  --epochs 2 \
  --lr 1e-4 \
  --num-generations 4 \
  --task-limit 30
```

### Full config reference

| Flag | Default | Description |
|------|---------|-------------|
| `--preset` | `t4` | Hardware preset: `t4`, `a100`, `v100`, `cpu` |
| `--env-url` | `http://localhost:8000` | Legacy option; current GRPO reward path uses local `PRReviewEnv` |
| `--output-dir` | `./grpo_checkpoint` | Where to save LoRA weights |
| `--epochs` | 2 | Training epochs |
| `--lr` | 2e-4 | Learning rate |
| `--num-generations` | 4 | GRPO rollouts per prompt (G) |
| `--task-limit` | 40 | Max tasks per epoch |
| `--lora-r` | 16 | LoRA rank |
| `--review-config` | `` | Path to review_config.json (optional) |
| `--wandb` | off | Set to log to Weights & Biases |

---

## 5. Expected Training Curves

A healthy run should show:

```
Epoch 1, Step 10 — mean_reward: 0.18, accuracy: 0.28
Epoch 1, Step 20 — mean_reward: 0.31, accuracy: 0.35
Epoch 1, Step 40 — mean_reward: 0.42, accuracy: 0.40
Epoch 2, Step 60 — mean_reward: 0.55, accuracy: 0.46
Epoch 2, Step 80 — mean_reward: 0.63, accuracy: 0.51
```

**Baselines to beat:**
- Heuristic: accuracy=0.385, mean_return=+0.166
- Random: accuracy=0.354, mean_return=-0.135

If mean_reward goes negative and stays there after step 20, something is wrong
(likely the env server is not responding or the model is outputting invalid JSON).

---

## 6. Monitoring

### Minimal (stdout)

The training script logs step-level reward and accuracy to stdout by default.

### Weights & Biases

```bash
pip install wandb
wandb login  # paste your API key

python train/grpo_train.py --preset a100 --wandb
```

This will log:
- `train/mean_reward` per step
- `train/accuracy` (correct verdicts / total)
- `train/loss`
- Tool call distribution (which tools the model favours)

### TensorBoard

```bash
pip install tensorboard
python train/grpo_train.py --preset a100 --report-to tensorboard
tensorboard --logdir ./grpo_checkpoint/runs
```

---

## 7. Checkpoint Management

### During training

Checkpoints save every `--save-steps` (default 50). The checkpoint folder:

```
grpo_checkpoint/
├── adapter_config.json     # LoRA config
├── adapter_model.safetensors
├── training_args.bin
└── checkpoint-50/
    ├── adapter_config.json
    └── adapter_model.safetensors
```

### Saving to Google Drive (Colab)

```python
from google.colab import drive
drive.mount("/content/drive")

import shutil
shutil.copytree("./grpo_checkpoint", "/content/drive/MyDrive/pr_review_grpo")
print("Saved.")
```

### Downloading from RunPod / Lambda

```bash
# From your local machine:
rsync -avz root@YOUR_RUNPOD_IP:/workspace/pr-review-agent/grpo_checkpoint/ ./grpo_checkpoint/
```

### Loading a checkpoint for inference

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", device_map="auto")
model = PeftModel.from_pretrained(base, "./grpo_checkpoint")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
```

---

## 8. Post-Training Evaluation

### Against the full benchmark

```bash
# Run evaluation
.venv/bin/python benchmarks/evaluate_trained_model.py \
  --checkpoint ./grpo_checkpoint \
  --output rewards/trained_eval.json
```

### Compare to baselines

```bash
PR_REVIEW_TOOL_BACKEND=heuristic \
.venv/bin/python benchmarks/evaluate_baselines.py \
  --output rewards/baseline_eval.json

.venv/bin/python - <<'PY'
import json
from pathlib import Path
for path in ["rewards/baseline_eval.json", "rewards/trained_eval.json"]:
    payload = json.loads(Path(path).read_text())
    print(path, payload.get("accuracy"), payload.get("mean_episode_return"))
PY
```

Expected output:

```
Policy          Tasks  Accuracy  Mean Return
-----------     -----  --------  -----------
heuristic          65     0.385        0.166
random             65     0.354       -0.135
grpo (trained)     65     0.XXX        X.XXX   ← target: >0.45 accuracy
```

---

## 9. Loader Benchmark (Manual, Post-Training)

This is a separate evaluation — not part of training. Run it after training is complete to measure whether injected org configuration changes review quality.

```bash
python benchmarks/loader_benchmark.py \
  --docs-dir docs/ \
  --checkpoint ./grpo_checkpoint \
  --output rewards/loader_benchmark.json
```

See `benchmarks/loader_benchmark.py` for what this measures.

---

## 10. Common Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| `CUDA out of memory` | Batch too large | Set `--preset t4` or reduce `--num-generations 2` |
| `GRPO requires at least 2 generations` | `num_generations` too low | Use `--num-generations 2` or higher |
| `generation_batch_size must be divisible by num_generations` | Incompatible TRL batch settings | Keep the script default `generation_batch_size=cfg.num_generations` |
| Model outputs non-JSON | System prompt not injected | Check `training_prompt()` and `build_obs_prompt()` in grpo_train.py |
| `ModuleNotFoundError: trl` | Deps not installed | `pip install trl peft bitsandbytes` |
| Training extremely slow | Sequential rollouts | Ensure `max_parallel_rollouts >= 2` in config |
| Reward goes to -0.55 and stays | Model always outputs same verdict | Increase exploration: `--lr 3e-4` or check reward shaping |
| `bitsandbytes` CUDA error | Old bitsandbytes | `pip install -U bitsandbytes` |

---

## 11. Colab Full Notebook Template

Save this as `notebooks/train_colab.ipynb` or paste cells in order:

```python
# Cell 1: Setup
!git clone https://github.com/YOUR_USERNAME/pr-review-agent.git
%cd pr-review-agent
!pip install -q trl peft bitsandbytes accelerate transformers datasets fastapi uvicorn httpx pyyaml fastmcp

# Cell 2: Train
!python train/grpo_train.py \
  --preset t4 \
  --output-dir ./grpo_checkpoint \
  --epochs 2 \
  --task-limit 40 \
  --num-generations 4

# Cell 3: Evaluate
!python benchmarks/evaluate_baselines.py --output rewards/baseline_eval.json
!python benchmarks/evaluate_trained_model.py \
  --checkpoint ./grpo_checkpoint \
  --output rewards/trained_eval.json

# Cell 4: Save to Drive
from google.colab import drive
drive.mount("/content/drive")
import shutil
shutil.copytree("./grpo_checkpoint", "/content/drive/MyDrive/pr_review_grpo", dirs_exist_ok=True)
print("Checkpoint saved to Drive")
```
