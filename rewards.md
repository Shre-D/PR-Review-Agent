```
# Reward Function Complete Specification — Detailed Summary

## Overview

This document summarizes a technical design conversation covering the complete specification of a reward function for training a policy model (via GRPO) to perform PR code review. The reward function trains the agent to **gather the right evidence before deciding**, rather than guessing efficiently. It is evaluated by a grader that operates independently from the agent's internal reasoning.

---

## Part 1: Foundational Framing

### Core Goal

The reward function has one job: make the policy learn that **gathering the right evidence before deciding is always worth more than guessing efficiently**. Every component either reinforces that or penalizes its violation.

### Grader Independence

The grader is epistemically independent of the agent's internal process. It receives only:

```python
GraderInput:
    task:              PRTask          # diff, risk_domains, expected_verdict, tier
    tool_results:      list[ToolResult] # all tools called this episode
    step_count:        int             # total steps including submit_review
    submitted_verdict: str             # "approve" | "reject" | "request_changes" | "escalate"
```

It does **not** receive:

- Router outputs
- SLM reasoning or chain-of-thought
- Confidence scores
- Any intermediate state

This is a **correctness requirement**, not a design preference. If the grader could read the SLM's self-reported confidence, the SLM could learn to write reasoning that games the grader without actually improving tool-calling behavior. The grader must be fully deterministic and auditable: same `GraderInput` → same reward, always.

---

## Part 2: Tier Scaffolding (Upstream of Reward)

Before any reward is computed, a **static tier router** constrains the episode shape. This defines what "valid" behavior looks like at each training stage. The tier sets the `decay_start` threshold that the efficiency penalty uses.

| Tier   | Task Size    | `decay_start` | `max_steps` |
| ------ | ------------ | --------------- | ------------- |
| Small  | Simple diff  | 2               | (small)       |
| Medium | Medium diff  | 4               | (medium)      |
| Large  | Complex diff | 6               | 9             |

The reward function is **not flat across task sizes**. A small task burning 5 steps is penalized harder than a large task using 5 steps, because their `decay_start` values differ. The tier label teaches task-size awareness through the reward structure, without requiring the agent to explicitly classify task complexity.

---

## Part 3: Component 1 — Per-Step Reward

Emitted every time the agent calls a tool that is **not** `submit_review` or `escalate`.

### Formula

```python
def step_reward(task, tool_name, tools_already_called, step_count) -> float:

    if tool_name in {"submit_review", "escalate"}:
        return 0.0  # terminal, handled separately

    if tool_name in tools_already_called:
        return -0.4  # duplicate: clearly worse than any novel tool

    reward = 0.1  # base: any novel tool call has value

    if tool_name in relevant_tools(task):
        reward += 0.2  # relevance bonus: covers the risk domain

    decay_start = tier_decay_start(task.tier)  # 2 / 4 / 6
    if step_count > decay_start:
        reward -= 0.05 * (step_count - decay_start)

    return round(max(-0.1, reward), 3)
```

### Explanation of Each Value

| Signal                   | Value                               | Rationale                                                                                                                                                                                            |
| ------------------------ | ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Base reward (novel call) | `+0.1`                            | Non-zero base ensures tool calls are always preferable to doing nothing. Even a sanity-check tool on a clean diff has value.                                                                         |
| Relevance bonus          | `+0.2`                            | The primary shaping force at step level. The gap between `+0.3` (relevant) and `+0.1` (irrelevant) is `0.2` — wide enough for GRPO to compute meaningful advantages.                             |
| Duplicate penalty        | `−0.4`                           | Clearly worse than any novel call. The gap from relevant (`+0.3`) to duplicate (`−0.4`) is `0.7`, ensuring repeating is never preferable.                                                         |
| Efficiency decay         | `−0.05 × (step − decay_start)` | Cumulative and tier-aware. On small tasks (`decay_start=2`), step 3 costs `−0.05`, step 4 costs `−0.10`, etc. On large tasks (`decay_start=6`), those same step numbers are free.             |

### What Per-Step Reward Does NOT Do

It does **not** evaluate the quality of what the tool returned. A `check_security` call returning zero findings earns the same `+0.13` as one returning three criticals. Quality of tool output is the grader's job at episode end. Conflating them at the step level would create a partially stochastic reward (tool output depends on diff content, not just agent choice), destabilizing learning.

---

## Part 4: Component 2 — Terminal Reward

Emitted once when the agent calls `submit_review` or `escalate`. This is the **dominant signal** — roughly 3–4× the magnitude of typical accumulated step rewards.

### Formula

```python
def terminal_reward(task, submitted_verdict, tool_results,
                    min_tools=None) -> float:

    correct = submitted_verdict.lower() == task.expected_verdict.lower()
    relevant_used = set(tool_results.keys()) & relevant_tools(task)
    early_submit = len(tool_results) < min_tools

    # Tier 1: correct verdict, sufficient evidence, relevant tools
    if correct and not early_submit and len(relevant_used) >= 1:
        support_bonus = min(0.2, 0.1 * max(0, len(relevant_used) - 1))
        return 1.0 + support_bonus + evidence_alignment_bonus(...)  # up to ~1.4

    # Tier 2: correct verdict, early submit (insufficient evidence)
    if correct and early_submit:
        return 0.2  # reachable but much worse than evidence-backed

    # Tier 3: correct verdict, no supportive relevant tool
    if correct:
        return -0.3

    # Tier 4: escalated when should have been reject/request_changes
    if (submitted_verdict == "escalate"
            and task.expected_verdict in {"reject", "request_changes"}):
        return -0.4 if early_submit else -0.2

    # Tier 5: wrong verdict
    return -1.0 if early_submit else -0.8
```

### Tier Boundary Rationale

| Tier                                   | Value                 | Reasoning                                                                                                                                                                                          |
| -------------------------------------- | --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Correct + relevant evidence            | `+1.0` to `+1.4`  | The ideal outcome. GRPO will exploit this gap.                                                                                                                                                     |
| Correct + early submit                 | `+0.2`              | Positive but low. The model can reach this outcome without being destroyed, but it's clearly worse than gathering evidence. Previous iterations used `−2.55`, which made the model afraid to ever submit. |
| Correct + no relevant evidence         | `−0.3`             | The model gathered enough tools to pass `min_tools` but none were relevant. Worse than early submit because the model wasted steps.                                                                |
| Escalate on genuinely ambiguous task   | `−0.2` to `−0.4` | Only applies when expected verdict was `reject` or `request_changes`. Escalation is not a safe default.                                                                                            |
| Wrong verdict                          | `−0.8` to `−1.0` | Strong negative signal. Early wrong is slightly worse than wrong with evidence.                                                                                                                    |

### The +0.8 Gap: The Most Important Design Decision

The gap between Tier 1 minimum (`+1.0`) and Tier 2 (`+0.2`) is **+0.8**. This means the expected return for gathering relevant evidence before a correct verdict is roughly **5× the return** for guessing correctly without it. GRPO will find and exploit this gap — which is exactly the intended behavior.

### Evidence Alignment Bonus

The `min(0.25, evidence × 0.25)` bonus is computed via a verdict-score alignment formula, not raw tool scores:

```python
def evidence_alignment_bonus(tool_results, verdict) -> float:
    score = aggregate_tool_scores(tool_results)
    # score near 0 = many problems found; score near 1 = clean
    if verdict == "approve":
        alignment = score          # high score + approve = aligned
    elif verdict in {"reject", "request_changes"}:
        alignment = 1.0 - score   # low score + reject = aligned
    else:
        alignment = 0.5           # escalate: neutral
    return min(0.25, alignment * 0.25)
```

A task where all tools returned high-severity findings that correctly predicted a `reject` gets a **high** alignment bonus, even though the raw `aggregate_tool_scores` is low — because the evidence **matched** the conclusion. This distinction (alignment vs. raw score) is critical.

### `aggregate_tool_scores` Helper

```python
def aggregate_tool_scores(tool_results) -> float:
    total_weight = 0
    weighted_sum = 0
    for tool_name, result in tool_results.items():
        weight = 1.5 if tool_name in relevant_tools(task) else 1.0
        weighted_sum += result.score * weight
        total_weight += weight
    return weighted_sum / total_weight if total_weight > 0 else 0.0
```

Relevant tools are weighted 1.5× over irrelevant tools.

---

## Part 5: Component 3 — Evidence Penalties (Grader Pass)

Applied at episode end by the grader, **separate from** the terminal reward tiers. These catch specific logical contradictions between tool outputs and the submitted verdict.

### Formula

```python
def evidence_penalties(task, tool_results, submitted_verdict) -> float:
    penalty = 0.0

    # Hard contradiction: critical finding + approve
    total_critical = sum(r.severity_counts.get("critical", 0)
                         for r in tool_results.values())
    if total_critical > 0 and submitted_verdict == "approve":
        penalty -= 0.40  # non-negotiable: criticals cannot be approved

    # Soft contradiction: many warnings + approve
    total_warnings = sum(r.severity_counts.get("warning", 0)
                         for r in tool_results.values())
    if total_warnings > 3 and submitted_verdict == "approve":
        penalty -= 0.15

    # Underweighting severity: 2+ criticals should be reject, not request_changes
    if total_critical >= 2 and submitted_verdict == "request_changes":
        penalty -= 0.10

    # False positive pattern: all tools clean + reject
    all_clean = all(r.score > 0.85 for r in tool_results.values())
    if all_clean and submitted_verdict == "reject":
        penalty -= 0.20  # teaches agent not to reject without evidence

    return penalty
```

### Penalty Rationale

| Condition                         | Penalty    | Rationale                                                                                                                                                                                                     |
| --------------------------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Critical findings + approve       | `−0.40` | Hardest constraint in the system. Applied even on top of `−0.55` wrong verdict, for a maximum terminal signal of `−0.95`. In real deployment, a missed critical is far more costly than a false reject. |
| 3+ warnings + approve             | `−0.15` | Soft contradiction. Not as severe as missing criticals, but approving with many warnings is structurally problematic.                                                                                         |
| 2+ criticals +`request_changes` | `−0.10` | Underweighting severity. Two criticals should trigger `reject`, not a softer response.                                                                                                                      |
| All tools clean + reject          | `−0.20` | False positive penalty. Without this, the agent could learn a conservative "when in doubt, reject" strategy. Groundless rejections are penalized symmetrically to missed criticals.                           |

---

## Part 6: Complete Episode Return

```python
episode_return = (
    sum(step_reward(task, tool, called_so_far, step_i)
        for tool, step_i in episode_steps)
    + terminal_reward(task, submitted_verdict, tool_results)
    + evidence_penalties(task, tool_results, submitted_verdict)
)
```

The ratio between an ideal episode and a wasteful one is approximately **3.5×**. GRPO's group-relative advantage normalization sees this as a large positive advantage for efficient evidence-gathering and a near-zero advantage for wasteful patterns.

---

## Part 7: Worst-Case Analysis and Raw Reward Scale

### Finding the True Theoretical Floor

The worst possible episode involves: calling one tool once (novel, `+0.1`), then duplicating it. Because duplicate penalty and efficiency decay **both** apply to the same steps, the reward can go very negative without a step cap. The practical floor is determined by `max_steps` per tier.

### Worst Case (Large Task, `max_steps=9`, `decay_start=6`)

| Steps              | Description                         | Reward                      |
| ------------------ | ----------------------------------- | --------------------------- |
| Step 1             | 1 novel irrelevant call             | `+0.1`                    |
| Steps 2–6         | 5 duplicates                        | `5 × −0.4 = −2.0`      |
| Step 7             | duplicate + decay `(7−6)`        | `−0.4 − 0.05 = −0.45`  |
| Step 8             | duplicate + decay `(8−6)`        | `−0.4 − 0.10 = −0.50`  |
| Step 9             | duplicate + decay `(9−6)`        | `−0.4 − 0.15 = −0.55`  |
| Terminal           | wrong verdict                       | `−1.0`                   |
| Evidence penalties | critical+approve + warnings+approve | `−0.40 − 0.15 = −0.55` |

**Full worst case:** `0.1 − 3.5 − 1.0 − 0.55 = −4.95` (clamped to `RAW_MIN = −1.5`)

### Best Case (Small Task, `decay_start=2`)

- 2 relevant novel calls within `decay_start`: `2 × 0.3 = +0.6`
- Correct verdict with evidence: `+1.0` + support bonus `+0.1` + alignment `+0.25`
- Zero penalties

**Best case:** `+1.95` (clamped to `RAW_MAX = +1.5`)

### Raw Reward Constants

```python
RAW_MIN = -1.5   # floor for GRPO
RAW_MAX =  1.5   # ceiling for GRPO
RAW_FLOOR = RAW_MIN        # malformed/error outputs
RAW_NEAR_FLOOR = -1.0      # tool errors, redirects
```

### Why No Normalization for Training

Previous iterations normalized raw rewards into `[0.01, 0.99]`:

```python
normalized = 0.01 + (raw - RAW_MIN) / (RAW_MAX - RAW_MIN) * 0.98
```

This compressed the gap between a relevant tool call and an irrelevant one from `0.2` raw to ~`0.018` normalized. GRPO could not compute meaningful advantages from that signal, and reward variance collapsed to zero (mode collapse). The model converged to a single action and stopped learning.

**Raw rewards go directly to GRPO.** `normalize_reward()` still exists for display purposes (UI, logs) but is not in the training path.

### Key Raw Reward Landmarks

| Scenario                           | Raw Reward |
| ---------------------------------- | ---------- |
| Evidence-backed correct submit     | `+1.0` to `+1.4` |
| Novel relevant tool                | `+0.3`    |
| Early correct submit               | `+0.2`    |
| Novel irrelevant tool              | `+0.1`    |
| Escalate (high-risk task)          | `−0.2`   |
| Correct, no supportive tool        | `−0.3`   |
| Duplicate tool                     | `−0.4`   |
| Wrong verdict (evidence)           | `−0.8`   |
| Wrong verdict (early)              | `−1.0`   |
| Malformed / error                  | `−1.5`   |

The total spread is **~2.5** between best and worst training outcomes. GRPO sees this as clear advantage signal.

---

## Summary: Design Principles

1. **Grader independence is non-negotiable.** No internal agent state, reasoning, or confidence scores are visible to the grader. This prevents gaming.
2. **Evidence-gathering must dominate guessing.** The `+0.8` gap between evidence-backed correct (`+1.0`) and early correct (`+0.2`) is the single most important number in the system.
3. **Raw rewards to GRPO.** No normalization in the training path. The reward scale is wide enough for GRPO to compute meaningful advantages.
4. **Escalation is not a safe default.** It only earns partial credit when the task was genuinely ambiguous (expected `reject`/`request_changes`). Escalating on clean diffs is penalized identically to wrong verdicts.
5. **Both false negatives and false positives are penalized.** Approving critical findings (`−0.40`) and rejecting clean code (`−0.20`) are symmetric failure modes that the reward function explicitly addresses.
6. **Tier-awareness is baked into reward structure.** The agent learns task-size sensitivity through `decay_start` differences, not through an explicit classification step.
7. **Early submit is reachable, not catastrophic.** `+0.2` for early correct submit lets the model learn to submit instead of looping forever, while still strongly incentivizing evidence collection.

```

```
