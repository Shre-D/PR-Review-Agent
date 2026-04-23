# GRPO Training Skill (TRL + OpenEnv)

## Official pattern (from OpenEnv tutorial 04-training.md)

```python
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoTokenizer
from envs.pr_review_env.client.pr_review_env_client import PRReviewEnv

model_name = "Qwen/Qwen3-1.7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Connect to a running environment (local or HF Space)
env_url = "http://localhost:8000"  # local dev
# env_url = "https://USER-pr-review-env.hf.space"  # deployed

# The rollout function — TRL will call this for each training prompt
async def rollout(prompts, model, tokenizer, **kwargs):
    trajectories = []
    async with PRReviewEnv(base_url=env_url) as env:
        for prompt in prompts:
            obs = await env.reset()
            messages = [{"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": obs.observation.diff_str}]
            total_reward = 0.0
            for step in range(8):  # max episode length
                # Generate action from model
                inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
                output = model.generate(inputs, max_new_tokens=64)
                action_text = tokenizer.decode(output[0], skip_special_tokens=True)
                action = parse_action(action_text)
                # Step environment
                result = await env.step(action)
                total_reward += result.reward
                if result.observation.done:
                    break
                messages.append({"role": "user", "content": str(result.observation)})
            trajectories.append({"prompt": prompt, "response": ..., "reward": total_reward})
    return trajectories
```

## Key hyperparameters

```python
config = GRPOConfig(
    output_dir="./grpo_checkpoint",
    num_train_epochs=3,
    per_device_train_batch_size=2,   # Qwen3-1.7B needs this low on T4
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    num_generations=8,               # G — rollouts per prompt for GRPO
    max_new_tokens=128,
    logging_steps=1,
)
```

## Installation (critical — use main branch TRL)

```bash
pip install git+https://github.com/huggingface/trl.git
pip install git+https://github.com/meta-pytorch/OpenEnv.git
pip install trackio vllm==0.10.2 bitsandbytes
```

## Common failure modes

1. **Flat reward curve** — reward too sparse. Add per-step rewards for each informative tool call.
2. **OOM on T4** — reduce batch to 1, increase grad_accum to 8, or use unsloth LoRA.
3. **Environment timeouts** — HF Space cold-starts. Duplicate the Space to your own account, or run locally.
4. **Action parse errors** — your model output doesn't match the action schema. Tighten system prompt with a JSON schema example.
5. **Training diverges** — learning rate too high. Try 5e-6.

## Reference notebooks
- `tutorial/examples/wordle.py` — GRPO + Wordle environment
- `tutorial/examples/unsloth_2048.ipynb` — Unsloth LoRA + 2048
- `examples/grpo_blackjack/` — full TRL + OpenEnv BlackJack training
