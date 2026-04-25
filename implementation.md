# Implementation Plan: Context-Aware PR Review RL Agent

## Purpose

This project should be positioned as a cheap, bounded, context-aware PR review
agent. Its job is not to replace a senior reviewer or prove correctness. Its job
is to reduce review and CI cost by deciding which evidence to gather, when to
stop gathering evidence, and how to turn that evidence into a preliminary review
verdict.

The useful real-world workflow is:

1. A PR arrives.
2. The environment builds a compact observation from the diff, changed files,
   scoped repo context, author/risk metadata, and available tools.
3. A small policy model chooses the next review action.
4. Tools produce deterministic evidence: static checks, config checks, build or
   type checks, test coverage signals, and repo-specific custom rules.
5. The policy decides whether more evidence is needed or whether it should
   submit `approve`, `request_changes`, `reject`, or `escalate`.
6. A human reviewer receives a cheap evidence-backed triage result.

The agent is useful when it saves expensive checks or reviewer time while
keeping high-risk PRs from being rubber-stamped. It is not useful if it simply
loops tools, guesses verdicts from labels, or claims broad code understanding
without evidence.

## Current Reality

The repository already has the core pieces:

- `PRReviewEnv` implements an OpenEnv-compatible environment.
- `tools.py` exposes review tools: security, quality, build/types, tests,
  config, submit, and escalate.
- `context_loader.py` extracts structural repo context from docs.
- `tasks.py` defines deterministic benchmark tasks.
- `adaptive_router.py` computes route tier, minimum evidence tools, and maximum
  review steps.
- `grpo_train.py` builds replayable training states and scores completions
  online through the environment.
- `benchmarks/` contains baseline and trained-model evaluation paths.

The current implementation also has several issues that prevent strong RL
claims:

- `inference.py` repeats the first heuristic tool action because it recomputes
  `heuristic_policy(observation)` each step and always selects index zero.
- The route budget is prompt text, not an authoritative contract shared by
  training, inference, evaluation, and reward.
- Duplicate/no-op actions can look moderately positive after normalization.
  Raw duplicate reward is `0.0`, but `normalize_reward(0.0)` is around `0.649`.
- One-step GRPO reward currently makes a direct correct submit more attractive
  than an initial evidence tool. This encourages guessing if the model can infer
  the expected verdict from the prompt.
- `[END] score=1.0` in inference is binary correctness, not normalized reward.
  The log label is misleading.
- Context is useful but mostly structural in short training mode. It includes
  architecture summary, critical paths, domain priorities, and enabled tools;
  it does not yet prove semantic doc-grounded review unless tasks and rewards
  require that behavior.

## Product Claim We Can Defend

After the fixes below, the defensible claim is:

> This is a low-cost, context-aware PR review triage agent that learns to route
> bounded review actions over deterministic tools and compact repository context.
> It improves tool selection, stopping behavior, and evidence-backed verdicts
> on benchmarked PR review tasks.

The claim we should not make:

> This guarantees correct PR review on arbitrary real-world repositories.

Tooling can make the workflow deterministic and measurable. It cannot guarantee
that every bug is detected, that every repo doc is relevant, or that the SLM
will reason correctly outside the training and evaluation distribution.

## Target Behavior

The policy should learn:

- which tool to call next
- which tools are relevant to a PR domain
- how to avoid duplicate/no-op actions
- when enough evidence exists
- when to stop before the maximum step budget
- which verdict to submit
- when to escalate instead of guessing
- how repo context changes risk and required evidence depth

The policy should not learn:

- always use all available steps
- always call `check_security`
- submit immediately because the expected verdict is inferable
- treat clean diffs as requiring maximal evidence
- treat normalized mid-scale reward as success

## Route Budget Design

All tasks need an episode horizon, but not all tasks need the same horizon.
The horizon is a safety cap, not a target number of steps.

Use `adaptive_router.py` as the shared source of truth:

```text
small task  -> lower min_tools, lower max_steps
medium task -> moderate min_tools, moderate max_steps
large task  -> higher min_tools, higher max_steps
```

The model learns where to stop inside that budget. The environment enforces the
budget so bad policies cannot loop forever.

Required implementation:

1. Add a shared helper such as `review_requirements(obs, review_config)` that
   returns:
   - `tier`
   - `min_tools`
   - `max_steps`
   - `requires_security`
   - `remaining_steps`
   - `evidence_tools_called`
   - `relevant_tools_called`
2. Use this helper in:
   - prompt construction
   - `PRReviewEnv` metadata
   - `inference.py`
   - `benchmarks/evaluate_trained_model.py`
   - GRPO reward scoring
   - baseline/evaluation reporting
3. Clamp author-depth scaling to a defensible upper bound. Large tasks can have
   more room than small tasks, but a training benchmark should not silently turn
   into 12-14 steps unless that is intentional and evaluated.

## Reward Design

The reward should expose only positive values to GRPO and evaluation:

```text
0.01 <= exposed_reward <= 0.99
```

Internally, raw reward can still use penalties. The exposed value should be a
bounded normalized score. This gives clean logs while preserving ranking.

The important part is not just positivity. The important part is ordering.

Required reward inequalities:

```text
malformed action ~= floor
invalid tool ~= floor
duplicate tool < novel irrelevant tool
novel irrelevant tool < novel relevant tool
early unsupported submit < relevant evidence tool
unsupported correct submit < evidence-backed correct submit
extra tool after sufficient evidence < efficient submit
wrong verdict ~= floor
contradicting evidence verdict ~= low reward
over-budget episode ~= low reward
```

Specific changes:

1. Malformed completions should return the positive floor instead of `-0.75`.
2. Replaying an already terminal state should return the floor instead of
   `-0.50`.
3. Tool errors should return the floor or near-floor, not subtract after
   normalization.
4. Duplicate actions should not normalize to mid-scale. They should be near the
   floor.
5. Submitting before `min_tools` should be low unless the route explicitly
   allows early submit and evidence is sufficient.
6. Correct submit with relevant evidence should be high, but not represented as
   perfect task success.
7. Binary task correctness should be reported separately from reward.

The reward should not inspect model reasoning or self-reported confidence as a
source of truth. Confidence can be logged and used for inference-time gating,
but not as a core correctness signal. Otherwise the model can learn to game the
grader.

## Context Use

The context loader improves the tool because it lets the agent compare diffs to
repo-specific expectations. However, this only matters if the task bank and
reward require context-sensitive decisions.

Existing useful context:

- architecture summary
- critical paths
- domain priorities
- author-depth overrides
- enabled/planned external tools
- custom rules when configured

Needed additions:

1. Add explicit context-dependent task labels:

```json
{
  "context_requirements": [
    "payments_require_idempotency",
    "auth_must_use_middleware"
  ],
  "expected_evidence": [
    "architecture_summary",
    "check_security"
  ]
}
```

2. Add tasks where context changes the verdict:
   - payment route missing idempotency
   - auth endpoint bypassing middleware
   - direct DB writes violating service-layer architecture
   - critical path touched by junior author requiring more evidence
   - config change violating deployment docs
   - migration violating rollback policy
3. Add a context-alignment reward component:
   - Did the selected tools cover context-required domains?
   - Did the final verdict match context-specific invariants?
   - Did the agent escalate when context and tool evidence were insufficient?

Without these additions, GRPO can still learn tool routing, but it cannot be
credited as learning repo-aware review beyond shallow prompt conditioning.

## GRPO Training Design

GRPO can help if local reward preferences match desired episode behavior.

Expected improvements:

- action JSON validity
- tool selection
- duplicate avoidance
- stopping before max budget
- verdict selection from tool evidence
- route-aware evidence depth
- better use of `review_history`

Weak or unsupported expectations:

- deep arbitrary code understanding
- strong generalization to unseen repositories
- finding issues with no tool/context signal
- guaranteed correctness on real PRs

Recommended training sequence:

1. Supervised warm start:
   - Train on curated traces with valid tool calls.
   - Include both tool steps and terminal submits.
   - Include negative examples only in evaluation, not as target outputs.
2. GRPO next-action training:
   - Use replayed states from `build_training_state_rows`.
   - Score next actions through the environment.
   - Ensure reward inequalities are correct before training.
3. Rollout-based GRPO:
   - Once next-action behavior is stable, score short/full trajectories.
   - This teaches stopping and complete review strategy better than one-step
     reward alone.
4. Held-out evaluation:
   - Never evaluate only on the same task patterns used for training.
   - Split static-tool tasks, context-dependent tasks, clean tasks, and
     escalation tasks.

## Implementation Steps

### Phase 1: Fix Control Flow and Logging

Files:

- `inference.py`
- `benchmarks/evaluate_trained_model.py`
- tests for inference/evaluation behavior

Work:

1. Fix heuristic inference to consume planned actions once.
2. Replace hardcoded local step caps with route-derived `max_steps`.
3. Make forced submit explicit in logs.
4. Rename `[END] score=...` to distinguish:
   - `correct=0|1`
   - `terminal_reward=...`
   - `episode_return=...`
5. Add tests proving heuristic inference does not repeat the first tool.

Acceptance:

- `python inference.py --model heuristic --limit 3` should not emit eight
  repeated `check_security` calls.
- Small tasks should stop well before large-task budgets.

### Phase 2: Make Route Requirements Authoritative

Files:

- `train/adaptive_router.py`
- `envs/pr_review_env/server/pr_review_env.py`
- `train/grpo_train.py`
- `benchmarks/evaluate_trained_model.py`
- tests for route metadata

Work:

1. Add a shared route requirement helper.
2. Include route fields in observation metadata.
3. Include route fields in training rows.
4. Use route fields in prompt construction.
5. Use route fields in evaluation loop bounds.

Acceptance:

- Prompt, observation metadata, reward, and eval agree on `max_steps`.
- Tests prove different task types get different budgets.

### Phase 3: Fix Reward Shape

Files:

- `envs/pr_review_env/server/grader.py`
- `envs/pr_review_env/server/pr_review_env.py`
- `train/grpo_train.py`
- `tests/test_grader.py`
- `tests/test_training_reward.py`

Work:

1. Expose only bounded positive rewards.
2. Make malformed, invalid, duplicate, over-budget, and wrong-verdict outcomes
   near floor.
3. Add early-submit penalty based on route `min_tools`.
4. Reward evidence-backed correct submit over unsupported correct submit.
5. Reward efficient submit over unnecessary extra tool after evidence is
   sufficient.

Acceptance:

- All exposed rewards satisfy `0.01 <= reward <= 0.99`.
- Duplicate reward is lower than a novel relevant tool.
- Early submit is lower than evidence-backed submit.
- Initial correct submit no longer beats best initial evidence tool across the
  task bank.

### Phase 4: Strengthen Context-Dependent Tasks

Files:

- `tasks/`
- `tasks/build_task_bank.py`
- `envs/pr_review_env/server/tasks.py`
- `tests/test_tasks.py`
- possibly `docs/review-tool.md` and `docs/company-guidelines.md`

Work:

1. Add task metadata for context requirements and expected evidence.
2. Add tasks where docs/config change the correct verdict.
3. Preserve relevant custom rules in short mode when they apply to the task.
4. Add evaluation buckets for context-dependent tasks.

Acceptance:

- There are held-out context-dependent tasks.
- A tool-only baseline performs worse on those tasks than a context-aware
  policy.
- Reports show context-dependent accuracy separately.

### Phase 5: Evaluation and Reporting

Files:

- `benchmarks/evaluate_baselines.py`
- `benchmarks/evaluate_trained_model.py`
- `benchmarks/generate_report.py`
- `rewards/` artifacts

Metrics:

- verdict accuracy
- mean normalized episode return
- duplicate tool rate
- invalid action rate
- average steps by route tier
- early submit rate
- over-budget/incomplete rate
- evidence-backed verdict rate
- context-dependent accuracy
- escalation precision/recall if escalation is trained
- cost proxy: number and type of tools called

Acceptance:

- A trained model must beat random and untrained SLM on action validity,
  duplicate rate, and evidence-backed verdict rate.
- A trained model should beat heuristic only if the heuristic is not already
  using privileged labels or fixed hand-coded policy.
- If it does not beat heuristic accuracy, it can still be valuable if it is
  cheaper, more context-aware, or more general than the heuristic baseline.

### Phase 6: Product Output and Low-Effort Production UX

The RL environment is not enough for adoption. A developer or CI system needs a
small, auditable review packet. This phase turns the environment output into
something useful in a real PR workflow.

Files:

- `inference.py`
- `benchmarks/evaluate_baselines.py`
- `benchmarks/evaluate_trained_model.py`
- optionally `envs/pr_review_env/models.py`
- tests for packet construction and CLI output

Work:

1. Add a review packet builder.

   Suggested location:

   - `inference.py` for the first low-effort implementation, or
   - a new helper such as `envs/pr_review_env/server/review_packet.py` if it is
     reused by UI, CLI, and benchmarks.

   Suggested function:

   ```python
   def build_review_packet(task, observation, episode_return, model_name: str) -> dict:
       ...
   ```

   It should return:

   ```json
   {
     "task_id": "docker_root_user",
     "model": "heuristic",
     "expected_verdict": "request_changes",
     "predicted_verdict": "request_changes",
     "correct": true,
     "risk": "high",
     "route": {
       "tier": "medium",
       "min_tools": 3,
       "max_steps": 7,
       "steps_used": 3,
       "remaining_steps": 4
     },
     "reward": {
       "episode_return": 2.13,
       "terminal_reward": 0.91
     },
     "tool_cost": {
       "total_calls": 3,
       "unique_tools": 3,
       "duplicate_calls": 0
     },
     "tools_used": [
       "check_security",
       "check_config",
       "submit_review"
     ],
     "findings": [
       {
         "tool": "check_config",
         "severity": "warning",
         "message": "Container runs without an explicit non-root user."
       }
     ],
     "evidence_summary": "Config evidence found container hardening issues.",
     "escalation_reason": ""
   }
   ```

2. Add concise CLI modes to `inference.py`.

   Suggested flags:

   ```text
   --output-format trace|packet|jsonl
   --show-expected
   --hide-diff
   ```

   Behavior:

   - `trace`: keep existing `[START]`, `[STEP]`, `[END]` logs for OpenEnv-style
     debugging.
   - `packet`: print one concise human-readable packet per task.
   - `jsonl`: print one JSON packet per line for CI ingestion.

   Human-readable packet example:

   ```text
   PR: docker_root_user
   Model: heuristic
   Risk: high
   Route: medium, 3/7 steps
   Decision: request_changes
   Evidence:
   - check_config: Base image is unpinned; consider digest or narrower version pinning.
   - check_config: Container runs without an explicit non-root user.
   Tool cost: 3 calls, 0 duplicates
   Episode return: 2.13
   ```

3. Make `[END]` unambiguous in trace mode.

   Replace:

   ```text
   [END] score=1.0
   ```

   With:

   ```text
   [END] correct=1 episode_return=2.13 terminal_reward=0.91 steps=3
   ```

   If `--show-expected` is set, include:

   ```text
   expected=request_changes predicted=request_changes
   ```

   Do not show expected verdict by default in production-style output. It is a
   benchmark label, not a real-world field.

4. Add practical metrics to benchmark outputs.

   Extend baseline and trained-model result JSON with:

   ```json
   {
     "duplicate_tool_rate": 0.02,
     "invalid_action_rate": 0.01,
     "early_submit_rate": 0.08,
     "over_budget_rate": 0.0,
     "evidence_backed_verdict_rate": 0.91,
     "mean_steps_by_tier": {
       "small": 2.4,
       "medium": 3.8,
       "large": 5.2
     },
     "mean_tool_calls": 3.1,
     "mean_unique_tool_calls": 2.9
   }
   ```

   Definitions:

   - duplicate tool call: same non-terminal tool called more than once in one
     episode.
   - invalid action: malformed JSON, unknown tool, invalid tool arguments, or
     tool execution error.
   - early submit: `submit_review` before route `min_tools`, unless route says
     early submit is allowed.
   - over budget: model did not submit before route `max_steps`.
   - evidence-backed verdict: terminal verdict with at least one relevant
     non-terminal tool result.

5. Add low-effort custom rules for repo-policy checks.

   These should start as deterministic rules, not model-generated reasoning.
   They can be represented as `ReviewConfig.custom_rules` or as documented
   structural extraction from `docs/review-tool.md` / `docs/company-guidelines.md`.

   Initial useful rules:

   - `src/payments/` or payment route changes should mention or preserve
     idempotency handling.
   - `src/auth/` or auth route changes should preserve middleware-based auth and
     avoid logging credentials/tokens.
   - `db/migrations/` changes should include a rollback/reverse path or be
     flagged for request changes.
   - `infrastructure/` and Kubernetes changes should not remove resource
     limits, probes, or secret references.
   - `.github/workflows/` changes should not broaden write permissions without
     justification.
   - Dockerfile changes should prefer pinned base images and non-root users.

   Example custom rule shape:

   ```json
   {
     "domain": "quality",
     "pattern": "db/migrations/.*RunPython\\(",
     "severity": "warning",
     "message": "Migration includes RunPython; verify rollback/reverse_code is present."
   }
   ```

   Important: rules should produce evidence. They should not directly overwrite
   verdicts. The policy still learns how to use evidence, and the grader checks
   whether the verdict aligns with evidence.

6. Add acceptance tests for product output.

   Suggested tests:

   - packet contains verdict, tools, findings, route, cost, and reward fields.
   - packet does not expose `expected_verdict` unless benchmark mode requests it.
   - jsonl output emits one valid JSON object per task.
   - trace output uses `correct=...`, not ambiguous `score=...`.
   - duplicate count is correct for repeated tool calls.
   - evidence-backed verdict rate counts only relevant tool evidence.

Acceptance:

- `python inference.py --model heuristic --limit 3 --output-format packet`
  prints a concise review packet for each task.
- `python inference.py --model heuristic --limit 3 --output-format jsonl`
  emits valid JSONL suitable for CI ingestion.
- Evaluation artifacts contain duplicate, invalid, evidence, step, and cost
  metrics.
- Product output does not claim perfect correctness. It presents evidence,
  risk, and a recommendation.

### Phase 7: Low-Effort Rule and Data Improvements

This phase improves usefulness without requiring a new model training run.

Files:

- `docs/review-tool.md`
- `docs/company-guidelines.md`
- `envs/pr_review_env/server/context_loader.py`
- `envs/pr_review_env/server/tasks.py`
- `tasks/build_task_bank.py`
- tests for context extraction and custom rules

Work:

1. Preserve applicable `custom_rules` in `task_review_config(..., mode="short")`.

   Current short mode clears `custom_rules`. That keeps prompts small, but it
   also removes cheap repo-policy evidence. Keep only rules whose domain or file
   pattern can apply to the current task.

   Suggested filtering:

   - keep rule if `rule.domain in task.risk_domains`
   - keep rule if regex/pattern mentions a touched path prefix
   - cap to a small number, such as 5 rules per task

2. Add explicit task metadata for context-dependent cases.

   Extend `PRTask` with optional fields:

   ```python
   context_requirements: list[str] = field(default_factory=list)
   expected_evidence: list[str] = field(default_factory=list)
   ```

   Backward compatibility:

   - default empty lists
   - existing JSONL remains loadable

3. Add small paired tasks to prove context matters.

   Paired task design:

   - same or similar diff
   - different repo context or critical path
   - different expected evidence depth or verdict

   Examples:

   - generic service route change can approve after light checks
   - payment route change without idempotency should request changes
   - generic logging change can request changes for noise
   - auth logging credentials/tokens should reject

4. Add a context-specific evaluation bucket.

   Benchmark output should include:

   ```json
   {
     "context_dependent": {
       "episodes": 12,
       "accuracy": 0.75,
       "mean_episode_return": 0.81
     }
   }
   ```

5. Add a loader ablation benchmark.

   Compare:

   - no context
   - short structural context
   - full structural context
   - short context plus custom rules

   Acceptance criterion:

   - context should improve context-dependent task accuracy or evidence-backed
     verdict rate without causing a large false-positive spike on clean tasks.

Acceptance:

- Short mode still keeps prompts compact.
- Context-dependent tasks exist and are reported separately.
- There is at least one benchmark where context changes model/tool behavior.
- If context does not improve results, the report says so instead of claiming
  repo-aware review.

## What Would Prove Real Utility

The strongest proof is not a single mean reward number. The strongest proof is
a table like this:

```text
Policy             Accuracy  Dup Rate  Avg Steps  Evidence Verdict  Context Acc  Cost
random             low       high      unstable   low               low          variable
heuristic          medium    low       fixed      medium            low/medium   fixed
base SLM           medium    high      unstable   medium            medium       variable
SFT SLM            higher    lower     stable     higher            medium       variable
SFT+GRPO SLM       higher    low       efficient  high              high         lower
```

For a real software engineering workflow, the agent earns its keep if it:

- catches high-risk PRs that cheap static checks can identify
- avoids expensive checks for obvious low-risk PRs
- highlights repo-specific policy violations
- produces auditable evidence for humans
- reduces duplicate/no-op tool calls
- escalates instead of guessing when evidence is insufficient

## Risks and Mitigations

Risk: The model learns task labels or benchmark shortcuts.

- Mitigation: use held-out tasks, mutate file names/descriptions, and evaluate
  context-dependent cases separately.

Risk: The reward trains direct guessing.

- Mitigation: lower unsupported submit reward and require evidence before high
  terminal reward.

Risk: Context is decorative rather than causal.

- Mitigation: add tasks where the same diff has different verdicts under
  different repo policies.

Risk: Tool outputs are too weak.

- Mitigation: integrate real analyzers where cheap, keep heuristic backend for
  deterministic tests, and report which backend was used.

Risk: The agent overuses tools to accumulate reward.

- Mitigation: duplicate floor, efficiency decay, route max steps, and submit
  reward once enough evidence exists.

Risk: GRPO instability on tiny data.

- Mitigation: SFT warm start, small learning rate sweeps, held-out eval, and
  compare to untrained and heuristic baselines.

## Final Recommendation

Continue, but keep the claim narrow and measurable.

This can become a credible cheap PR review agent if the implementation focuses
on bounded evidence gathering and context-aware triage. The project should not
claim general code review correctness. It should claim a measurable improvement
in review workflow policy: fewer invalid actions, fewer duplicate tools, better
tool routing, earlier stopping on low-risk PRs, and higher evidence-backed
verdict accuracy on held-out benchmark tasks.
