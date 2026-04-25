#!/usr/bin/env bash
# End-to-end reproducibility receipt: install, test, exercise the training
# reward path on a 3-task CPU slice. Should finish in <3 minutes.
#
# Run from the repository root:
#   bash scripts/smoke.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "== install (.[server,train,dev]) =="
python -m pip install -q -e ".[server,train,dev]"

echo
echo "== test suite (heuristic backend) =="
PR_REVIEW_TOOL_BACKEND=heuristic pytest -q

echo
echo "== task bank =="
python tasks/build_task_bank.py --print-summary

echo
echo "== training data + reward path (3 tasks, no model load) =="
PR_REVIEW_TOOL_BACKEND=heuristic python - <<'PY'
from train.train_config import TrainingConfig
from train.grpo_train import build_training_state_rows, score_completion_locally, TrainingMetricsCounter

cfg = TrainingConfig.for_cpu()
cfg.tasks_file = "all"
cfg.training_task_limit = 3

rows = build_training_state_rows(cfg)
print(f"  built {len(rows)} replayable training rows")

counter = TrainingMetricsCounter()
for row in rows[:5]:
    score_completion_locally(
        '{"tool_name":"check_security","arguments":{}}',
        row["task_id"],
        replay_actions=row["replay_actions"],
        counter=counter,
    )
print("  counter snapshot:", {k: v for k, v in counter.flush().items() if v})
PY

echo
echo "== baseline eval (random + heuristic on 78 tasks) =="
PR_REVIEW_TOOL_BACKEND=heuristic python benchmarks/evaluate_baselines.py \
  --task-bank all --task-loader-mode short \
  --output rewards/baseline_eval.json \
  >/dev/null
python - <<'PY'
import json
data = json.load(open("rewards/baseline_eval.json"))
for name, summary in data.items():
    print(f"  {name:<10}  acc={summary['accuracy']:.3f}  return={summary['mean_episode_return']:+.3f}")
PY

echo
echo "smoke ok"
