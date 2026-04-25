---
title: "GRPO: Teaching a 1.7B Model to Route Tool Calls"
date: 2026-04-24
tags: [grpo, trl, training, qwen, llm]
---

# GRPO: Teaching a 1.7B Model to Route Tool Calls

Group Relative Policy Optimization (GRPO) is the training algorithm that turns Qwen/Qwen3-1.7B from a general-purpose language model into a focused code review agent. Here's how it works and why we chose it.

## What GRPO Is

GRPO is a variant of PPO designed for language model fine-tuning. The key innovation: instead of computing a value function baseline (which requires a separate critic model), GRPO computes baselines by generating multiple completions per prompt and using their relative rewards.

For each training prompt, generate G completions. The reward for each completion is normalized relative to the group mean:

```
advantage_i = (reward_i - mean(rewards)) / std(rewards)
```

This works well for code review because:
1. **No value function needed**: One fewer neural network to train and maintain
2. **Natural variance**: Different tool routing strategies get different rewards, creating clear learning signal
3. **Group normalization**: The agent learns relative quality ("calling check_security first was better than check_quality first") rather than absolute quality

## The Action Schema

The agent outputs JSON at each step:

```json
{"tool_name": "check_security", "arguments": {"diff_str": "diff --git..."}}
```

For the final verdict:
```json
{"tool_name": "submit_review", "arguments": {"verdict": "reject", "reasoning": "SQL injection detected"}}
```

The system prompt tells the model exactly what to output:

```
You are a code reviewer. For each step, output a JSON tool call:
{"tool_name": "<name>", "arguments": {<args>}}

Available tools: check_security, check_quality, check_build_and_types,
check_tests, check_config, submit_review, escalate.
```

## The Rollout Function

```python
async def rollout_episode(env_client, model, tokenizer, device) -> dict:
    step_result = await env_client.reset()
    obs = step_result.observation
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_obs_prompt(obs)},
    ]
    total_reward = 0.0

    for _ in range(MAX_STEPS_PER_EPISODE):  # max 8 steps
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            output_ids = model.generate(input_ids, max_new_tokens=128, do_sample=False)
        response = tokenizer.decode(output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)

        action = parse_action(response)
        step_result = await env_client.step(action)
        total_reward += step_result.reward or 0.0

        messages.append({"role": "assistant", "content": response})
        if step_result.observation.done:
            break
        messages.append({"role": "user", "content": build_obs_prompt(step_result.observation)})

    return {"prompt": ..., "response": last_response, "reward": total_reward}
```

We use greedy decoding (`do_sample=False`) during rollout. Exploration happens implicitly through the diversity of prompts (different diffs, different languages, different risk profiles).

## Why Qwen/Qwen3-1.7B?

Several reasons:

1. **Size**: 1.7B parameters fits on a single T4 GPU with bfloat16 and still leaves room for gradient computation
2. **Instruction following**: Qwen3 is fine-tuned for chat/instruction tasks, meaning the base model already knows to follow JSON output schemas
3. **No proprietary deps**: Pure open-weights model, deployable anywhere
4. **Speed**: Small enough for real-time CI integration (< 1 second per action on GPU)

## Key Hyperparameters

```python
config = GRPOConfig(
    output_dir="./grpo_checkpoint",
    num_train_epochs=3,
    per_device_train_batch_size=2,   # 1 on T4, 2 on A100
    gradient_accumulation_steps=4,   # effective batch = 8
    learning_rate=1e-5,              # conservative for fine-tuning
    num_generations=8,               # G — rollouts per prompt
    max_new_tokens=128,              # JSON action fits easily in 128 tokens
    logging_steps=1,
)
```

The `num_generations=8` (G) is the GRPO-specific parameter. Higher G gives better baselines but costs more compute. At 8, we get meaningful variance in tool routing strategies within each batch.

## The Parse Fallback

Not every model output will be valid JSON. The `parse_action` function handles this gracefully:

```python
def parse_action(text: str) -> PRReviewAction:
    try:
        start = text.rfind("{")
        end = text.rfind("}") + 1
        payload = json.loads(text[start:end])
        return PRReviewAction(tool_name=payload["tool_name"], arguments=payload.get("arguments", {}))
    except Exception:
        return PRReviewAction(
            tool_name="submit_review",
            arguments={"verdict": "approve", "reasoning": "parse error fallback"},
        )
```

Parse failures receive the `approve` fallback — which typically gets a -0.55 terminal reward on tasks that should be rejected. This creates natural pressure for the model to learn to output valid JSON.

## What the Model Learns

After GRPO training, we expect the model to learn:

1. **Language routing**: Python diffs → `check_security` + `check_quality`. Dockerfiles → `check_config`. Go refactors → `check_build_and_types`.
2. **Efficiency**: Stop calling tools after 3-4 steps. The efficiency penalty teaches this.
3. **Evidence-first verdicts**: Never submit without calling at least one relevant tool.
4. **Duplicate avoidance**: The -0.20 penalty for repeated tool calls is the strongest per-step signal.

---

*Next: [Baselines and Evaluation: Measuring What Matters](./07-baselines.md)*
