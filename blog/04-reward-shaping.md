---
title: "Reward Shaping for Code Review: The Details"
date: 2026-04-24
tags: [reward-shaping, grpo, rl-training, grader]
---

# Reward Shaping for Code Review: The Details

Reward shaping is where most RL projects quietly fail. A reward function that's too sparse teaches the agent nothing. One that's too dense teaches the agent to exploit loopholes. Getting this right for code review took several design iterations.

## The Core Insight

A good code reviewer has two properties: they call the **right tools** and they emit the **right verdict**. Bad reviewers do one but not the other — they run every possible tool and still guess the verdict, or they guess without gathering evidence.

Our reward function encodes both properties explicitly.

## Per-Step Rewards

Every time the agent calls an analysis tool (not `submit_review` or `escalate`), it receives a step reward:

```python
def step_reward(task, tool_name, already_called, step_count) -> float:
    if tool_name in {"submit_review", "escalate"}:
        return 0.0  # handled by terminal_reward

    if already_called:
        return -0.20  # strong penalty for redundant calls

    reward = 0.05   # base reward for any new tool call
    if tool_name in relevant_tools(task):
        reward += 0.08  # bonus for calling a tool that matches risk_domains

    if step_count > 4:
        reward -= 0.03 * (step_count - 4)  # efficiency penalty

    return round(reward, 3)
```

The key numbers:
- **+0.05**: Any new tool call is worth something. We want the agent to gather at least some evidence.
- **+0.08 bonus**: Calling a tool relevant to the task's actual risk domains is better. `check_security` on a SQL injection task earns the bonus. `check_build_and_types` on that same task earns only the base.
- **-0.20**: Calling the same tool twice is harshly penalized. This prevents the degenerate strategy of calling `check_security` repeatedly.
- **-0.03 × (step - 4)**: After step 4, each additional step reduces the reward. Easy tasks should be resolved in 2-3 steps. Hard tasks might need 5-6. But no task needs 10.

## Relevant Tools

The `relevant_tools()` function maps each task's `risk_domains` to the tools that cover them:

```python
DOMAIN_TO_TOOLS = {
    "security": {"check_security"},
    "quality":  {"check_quality"},
    "tests":    {"check_tests"},
    "config":   {"check_config"},
    "build":    {"check_build_and_types", "check_config"},
}

def relevant_tools(task: PRTask) -> set[str]:
    if not task.risk_domains:
        return {"check_quality"}  # clean diffs still need a sanity check
    tools = set()
    for domain in task.risk_domains:
        tools |= DOMAIN_TO_TOOLS.get(domain, set())
    return tools
```

A task with `risk_domains=["security", "config"]` will reward `check_security` and `check_config` calls. A clean diff with `risk_domains=[]` rewards `check_quality` as a minimal baseline.

## Terminal Rewards

The terminal reward is the big signal — received once when the agent calls `submit_review` or `escalate`:

```python
def terminal_reward(task, submitted_verdict, tool_results) -> float:
    correct = submitted_verdict.lower() == task.expected_verdict.lower()
    evidence = aggregate_tool_scores(tool_results)  # 0-1 weighted average
    supportive = len(set(tool_results) & relevant_tools(task))

    if correct and supportive >= 1:
        return 1.0 + min(0.25, evidence * 0.25)  # up to 1.25

    if correct and supportive == 0:
        return 0.35  # right verdict, no evidence — lucky guess

    if submitted_verdict == "escalate" and task.expected_verdict in {"reject", "request_changes"}:
        return 0.10  # partial credit for appropriate uncertainty

    return -0.55  # wrong verdict — strong negative signal
```

The tier structure matters:

| Outcome | Reward |
|---------|--------|
| Correct + evidence + high tool scores | up to **+1.25** |
| Correct + at least one relevant tool | **+1.0** |
| Correct + no supporting evidence | **+0.35** |
| Escalated (was reject/request_changes) | **+0.10** |
| Wrong verdict | **-0.55** |

The gap between "+1.0 with evidence" and "+0.35 without" is the core training signal: the agent must learn that gathering evidence before concluding is always worth it.

## Why -0.55 for Wrong Verdicts?

We debated whether to use -1.0 here. The risk with a very large negative terminal reward is that the agent becomes risk-averse — it over-uses `escalate` to avoid the penalty. At -0.55, the penalty is substantial enough to matter but not so catastrophic that escalation always dominates.

## The Full Episode Return

A typical efficient episode looks like this:

```
Step 1: check_security  → +0.13 (base 0.05 + bonus 0.08)
Step 2: check_config    → +0.13 (base 0.05 + bonus 0.08)
Step 3: submit_review   → +1.0  (correct verdict, 2 relevant tools)
Total:                     +1.26
```

A wasteful episode:

```
Step 1: check_security  → +0.13
Step 2: check_security  → -0.20 (duplicate!)
Step 3: check_quality   → +0.05 (not relevant, no bonus)
Step 4: check_tests     → +0.05 (not relevant)
Step 5: check_build     → +0.02 (step penalty: 0.05 - 0.03)
Step 6: submit_review   → +0.35 (correct, but only one relevant tool used)
Total:                     +0.40
```

The reward function makes efficient, evidence-backed reviews worth 3× more than lucky but wasteful ones.

---

*Next: [Five Tools That Make Our Reviewer Smart](./05-tools.md)*
