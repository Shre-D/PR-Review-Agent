# Instructions 2: Final Hackathon Evidence Pass

Goal: turn the current benchmark + training path into a submission with one trained checkpoint/evaluation artifact and no stale rollout code path.

## 1. Repair Or Remove `rollout_episode`

Current issue: `train/grpo_train.py::rollout_episode()` sends model actions directly to the HTTP env. If the model emits an analysis tool call with empty `arguments`, the env injects `task_id` but not `diff_str`, so the tool call can fail.

Preferred fix:

1. Reuse `_action_with_state_args(action, obs)` inside `rollout_episode()` before calling `env_client.step()`.
2. Ensure terminal actions still get default `reasoning` and `confidence`.
3. Add a unit test with a fake client/model/tokenizer or a small direct helper test proving an empty `check_security` action becomes an action with `diff_str`.

Acceptance:

```bash
.venv/bin/python -m pytest -q tests/test_training_reward.py tests/test_action_parsing.py
```

## 2. Add Trained Checkpoint Evaluation Script

Create `benchmarks/evaluate_trained_model.py`.

Minimum behavior:

1. Accept:
   - `--checkpoint`
   - `--base-model`, default `Qwen/Qwen3-1.7B`
   - `--tasks-file`, default `tasks/tasks.jsonl`
   - `--limit`, default `0` meaning all tasks
   - `--output`, default `rewards/trained_eval.json`
   - `--review-config`, optional
2. Load base model + LoRA checkpoint if a checkpoint path is provided.
3. Run the same PRReviewEnv episodes against selected tasks.
4. For each step:
   - build prompt with `build_obs_prompt`
   - generate one JSON action
   - parse with `parse_action`
   - use `_action_with_state_args(action, obs)` before stepping env
   - stop when `obs.done` or `max_steps_per_episode` is reached
5. Write JSON summary:
   - `episodes`
   - `accuracy`
   - `mean_episode_return`
   - `examples`
   - per-language breakdown if time permits

Acceptance:

```bash
.venv/bin/python benchmarks/evaluate_trained_model.py \
  --checkpoint grpo_checkpoint \
  --limit 5 \
  --output /tmp/trained_eval_smoke.json
```

If no checkpoint exists yet, the script should fail clearly with a useful message.

## 3. Run A Small Training Job

Use the current online env reward loop. Start small first.

Recommended first smoke:

```bash
PR_REVIEW_TOOL_BACKEND=heuristic \
.venv/bin/python train/grpo_train.py \
  --preset cpu \
  --task-limit 3 \
  --epochs 1 \
  --output-dir ./grpo_checkpoint_smoke
```

If CPU model loading is too slow, move to GPU. Recommended T4/A100 command:

```bash
PR_REVIEW_TOOL_BACKEND=heuristic \
.venv/bin/python train/grpo_train.py \
  --preset t4 \
  --task-limit 20 \
  --epochs 1 \
  --num-generations 2 \
  --output-dir ./grpo_checkpoint
```

For final artifact, prefer:

```bash
PR_REVIEW_TOOL_BACKEND=heuristic \
.venv/bin/python train/grpo_train.py \
  --preset t4 \
  --task-limit 40 \
  --epochs 2 \
  --num-generations 4 \
  --output-dir ./grpo_checkpoint
```

Record:

- hardware
- command
- wall-clock time
- final training logs
- checkpoint directory contents

## 4. Evaluate The Checkpoint

Run:

```bash
PR_REVIEW_TOOL_BACKEND=heuristic \
.venv/bin/python benchmarks/evaluate_trained_model.py \
  --checkpoint ./grpo_checkpoint \
  --output rewards/trained_eval.json
```

Also refresh baselines:

```bash
PR_REVIEW_TOOL_BACKEND=heuristic \
.venv/bin/python benchmarks/evaluate_baselines.py \
  --output rewards/baseline_eval.json
```

Compare:

- random accuracy / mean return
- heuristic accuracy / mean return
- trained policy accuracy / mean return

If trained policy does not beat heuristic, still report it honestly as a first trained SLM policy and explain the next improvement: more tasks, longer run, stronger reward shaping, or SFT warm start from teacher traces.

## 5. Update Submission Docs

Update `README.md`:

1. Add a “Submission Results” table:

```markdown
| Policy | Episodes | Accuracy | Mean Return |
|--------|----------|----------|-------------|
| Random | ... | ... | ... |
| Heuristic | ... | ... | ... |
| Trained SLM | ... | ... | ... |
```

2. Add exact training command used.
3. Add exact evaluation command used.
4. State clearly that adaptive routing is SLM-only and route tiers control evidence depth, not model size.

Update `TRAINING.md`:

- Add the known-good command.
- Add checkpoint evaluation command.
- Add troubleshooting if GRPO rejects `num_generations < 2`.

Update `.gitignore` only if needed:

- Keep large checkpoints ignored.
- Keep small JSON artifacts unignored:
  - `rewards/baseline_eval.json`
  - `rewards/trained_eval.json`

## 6. Final Verification

Run:

```bash
.venv/bin/python -m pytest -q
PR_REVIEW_TOOL_BACKEND=heuristic .venv/bin/python benchmarks/evaluate_baselines.py
PR_REVIEW_TOOL_BACKEND=heuristic .venv/bin/python benchmarks/evaluate_trained_model.py --checkpoint ./grpo_checkpoint --limit 5 --output /tmp/trained_eval_smoke.json
```

Expected:

- tests pass
- baseline JSON writes
- trained eval JSON writes
- no path references missing files
- README claims match actual artifacts

## 7. Submission Framing

Use this wording if the trained policy is modest:

> This submission provides a working PR review routing benchmark and SLM-only GRPO training loop. The final demo includes random and heuristic baselines plus a first trained SLM checkpoint evaluated on the same task bank. The adaptive router does not call larger LLMs; it changes evidence requirements and tool budget only.

Avoid claiming:

- production GitHub bot
- trained model beats heuristic unless the artifact proves it
- LLM-based adaptive routing
- full multi-step on-policy rollout if training still uses replayed benchmark states
