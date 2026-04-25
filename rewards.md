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
        return -0.20  # duplicate: strong, unambiguous penalty

    reward = 0.05  # base: any novel tool call has value

    if tool_name in relevant_tools(task):
        reward += 0.08  # relevance bonus: covers the risk domain

    decay_start = tier_decay_start(task.tier)  # 2 / 4 / 6
    if step_count > decay_start:
        reward -= 0.03 * (step_count - decay_start)

    return round(reward, 3)
```

### Explanation of Each Value

| Signal                   | Value                               | Rationale                                                                                                                                                                                            |
| ------------------------ | ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Base reward (novel call) | `+0.05`                           | Zero base would make tool calls neutral on noisy episodes, risking the policy learning to skip evidence on easy tasks. Even a sanity-check tool on a clean diff is worth something.                  |
| Relevance bonus          | `+0.08`                           | The primary shaping force at step level. The gap between `+0.13` (relevant) and `+0.05` (irrelevant) is the training signal that makes domain-awareness emerge.                                  |
| Duplicate penalty        | `−0.20`                          | Deliberately harsh. Must exceed two base rewards combined (`2 × 0.05 = 0.10`). Ensures repeating is never preferable to doing something new, even on noisy episodes.                              |
| Efficiency decay         | `−0.03 × (step − decay_start)` | Cumulative and tier-aware. On small tasks (`decay_start=2`), step 3 costs `−0.03`, step 4 costs `−0.06`, etc. On large tasks (`decay_start=6`), those same absolute step numbers are free. |

### What Per-Step Reward Does NOT Do

It does **not** evaluate the quality of what the tool returned. A `check_security` call returning zero findings earns the same `+0.13` as one returning three criticals. Quality of tool output is the grader's job at episode end. Conflating them at the step level would create a partially stochastic reward (tool output depends on diff content, not just agent choice), destabilizing learning.

---

## Part 4: Component 2 — Terminal Reward

Emitted once when the agent calls `submit_review` or `escalate`. This is the **dominant signal** — roughly 3–4× the magnitude of typical accumulated step rewards.

### Formula

```python
def terminal_reward(task, submitted_verdict, tool_results) -> float:

    correct = submitted_verdict.lower() == task.expected_verdict.lower()
    relevant_used = set(tool_results.keys()) & relevant_tools(task)
    evidence = aggregate_tool_scores(tool_results)  # 0–1 weighted average

    # Tier 1: correct verdict, supported by relevant evidence
    if correct and len(relevant_used) >= 1:
        return 1.0 + min(0.25, evidence * 0.25)  # up to 1.25

    # Tier 2: correct verdict, no domain-relevant tool called
    if correct and len(relevant_used) == 0:
        return 0.35  # right answer, lucky guess

    # Tier 3: escalated when should have been reject/request_changes
    if (submitted_verdict == "escalate"
            and task.expected_verdict in {"reject", "request_changes"}):
        return 0.10  # appropriate uncertainty, partial credit

    # Tier 4: wrong verdict
    return -0.55
```

### Tier Boundary Rationale

| Tier                                 | Value                 | Reasoning                                                                                                                                                                                                                                                                                                               |
| ------------------------------------ | --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Correct + relevant evidence          | `+1.0` to `+1.25` | The ideal outcome. GRPO will exploit this gap.                                                                                                                                                                                                                                                                          |
| Correct + no relevant evidence       | `+0.35`             | Not zero or negative — on easy tasks, the correct verdict really is obvious from the diff. Punishing correct answers would teach the agent to call irrelevant tools just to avoid the penalty.`+0.35` says "right answer, but no demonstrated process."                                                              |
| Escalate on genuinely ambiguous task | `+0.10`             | Only applies when expected verdict was `reject` or `request_changes`. Escalation on a clean diff that should be approved earns `−0.55`. Escalation is not a safe default.                                                                                                                                        |
| Wrong verdict                        | `−0.55`            | Not `−1.0`. At `−1.0`, expected value of guessing on a 50/50 uncertain task becomes `0.5×1.0 − 0.5×1.0 = 0.0`, making escalation always dominate. At `−0.55`, expected value of guessing = `0.5×1.0 − 0.5×0.55 = +0.225`, making guessing still worth attempting when evidence is reasonably strong. |

### The +0.65 Gap: The Most Important Design Decision

The gap between Tier 1 minimum (`+1.0`) and Tier 2 (`+0.35`) is **+0.65**. This means the expected return for gathering relevant evidence before a correct verdict is roughly **2× the return** for guessing correctly without it. GRPO will find and exploit this gap — which is exactly the intended behavior.

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

## Part 7: Worst-Case Analysis and Reward Normalization

### Finding the True Theoretical Floor

The conversation identified that the worst possible episode involves: calling one tool once (novel, `+0.05`), then duplicating it forever. Because duplicate penalty and efficiency decay **both** apply to the same steps and stack, the reward diverges to `−∞` without a step cap. The practical floor is therefore determined by `max_steps` per tier.

### Worst Case (Large Task, `max_steps=9`, `decay_start=6`)

| Steps              | Description                         | Reward                      |
| ------------------ | ----------------------------------- | --------------------------- |
| Step 1             | 1 novel irrelevant call             | `+0.05`                   |
| Steps 2–6         | 5 duplicates                        | `5 × −0.20 = −1.00`    |
| Step 7             | duplicate + decay `(7−6)`        | `−0.20 − 0.03 = −0.23` |
| Step 8             | duplicate + decay `(8−6)`        | `−0.20 − 0.06 = −0.26` |
| Step 9             | duplicate + decay `(9−6)`        | `−0.20 − 0.09 = −0.29` |
| Terminal           | wrong verdict                       | `−0.55`                  |
| Evidence penalties | critical+approve + warnings+approve | `−0.40 − 0.15 = −0.55` |

**Total step penalties:** `−1.78`
**Full worst case:** `0.05 − 1.78 − 0.55 − 0.55 = −2.83`

### Best Case (Small Task, `decay_start=2`)

- 2 relevant novel calls within `decay_start`
- Correct verdict
- Maximum evidence alignment bonus
- Zero penalties

**Best case:** `+1.51`

### Normalization Formula

```
RAW_MIN = -2.83
RAW_MAX =  1.51
span    =  4.34

normalized = 0.01 + (raw + 2.83) / 4.34 × 0.98
```

```python
RAW_MIN = -2.83  # large task, max_steps=9, 8 duplicates after step 1,
                 # wrong verdict, critical+approve + warnings+approve penalties,
                 # efficiency decay on steps 7-9 stacked with duplicate penalty
RAW_MAX =  1.51  # small task, 2 relevant novel calls, correct verdict,
                 # max evidence alignment bonus, zero penalties

def normalize_reward(raw: float) -> float:
    clipped = max(RAW_MIN, min(RAW_MAX, raw))
    return round(0.01 + (clipped + 2.83) / 4.34 * 0.98, 4)
```

### Key Observations on Normalized Space

| Scenario                            | Raw Score  | Normalized Score | Notes                                                                     |
| ----------------------------------- | ---------- | ---------------- | ------------------------------------------------------------------------- |
| Ideal episode                       | `+1.51`  | ~`0.99`        | Best case                                                                 |
| Correct verdict, evidence, no bonus | `+1.00`  | ~`0.86`        | Good                                                                      |
| Wrong verdict, no penalties         | `−0.55` | ~`0.57`        | "Clean miss"                                                              |
| Correct, lucky guess (no evidence)  | `+0.35`  | ~`0.74`        | Above midpoint but below evidenced correct                                |
| Worst case                          | `−2.83` | `0.01`         | Floor                                                                     |
| Normalized midpoint                 | `0.50`   | →               | Corresponds to raw score of ~`−0.73` (within the wrong-verdict region) |

**The `0.50` midpoint** sits comfortably inside the wrong-verdict region, meaning the semantic split holds: anything above `0.50` is at least partially acceptable behavior; anything below is genuinely bad.

**The wrong-verdict-no-penalty case landing at `0.57`** (above `0.50`) is intentional. A wrong verdict with no supporting tool calls and no evidence contradictions is a "clean miss" — the agent didn't do anything structurally wrong, it just got the answer wrong. The gap between `0.57` (wrong) and `0.74` (correct lucky guess) is the gradient that teaches evidence-gathering.

> **Note:** If wrong verdicts always sitting below `0.50` is desired regardless of penalties, `RAW_MIN` would need to shift to around `−1.10`. However, this would compress the catastrophic failure region and reduce gradient signal there, which is an explicit trade-off.

---

## Summary: Design Principles

1. **Grader independence is non-negotiable.** No internal agent state, reasoning, or confidence scores are visible to the grader. This prevents gaming.
2. **Evidence-gathering must dominate guessing.** The `+0.65` gap between evidenced correct and lucky correct is the single most important number in the system.
3. **Escalation is not a safe default.** It only earns partial credit when the task was genuinely ambiguous (expected `reject`/`request_changes`). Escalating on clean diffs is penalized identically to wrong verdicts.
4. **Both false negatives and false positives are penalized.** Approving critical findings (`−0.40`) and rejecting clean code (`−0.20`) are symmetric failure modes that the reward function explicitly addresses.
5. **Tier-awareness is baked into reward structure.** The agent learns task-size sensitivity through `decay_start` differences, not through an explicit classification step.
6. **Normalization preserves gradient structure.** The `[0.01, 0.99]` normalized range reflects genuine behavioral quality, with `0.50` falling semantically at the boundary between acceptable and bad behavior.

```

```
