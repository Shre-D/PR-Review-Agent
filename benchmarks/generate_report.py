"""Generate a visual training report from benchmark artifacts.

Reads:
  rewards/baseline_eval.json   — random + heuristic results
  rewards/trained_eval.json    — trained SLM results
  grpo_checkpoint/training_log.csv  — per-step loss/reward from RewardLogCallback

Writes:
  rewards/comparison_report.png  — 4-panel figure

Usage:
  python benchmarks/generate_report.py
  python benchmarks/generate_report.py \\
      --baseline rewards/baseline_eval.json \\
      --trained  rewards/trained_eval.json \\
      --log      grpo_checkpoint/training_log.csv \\
      --output   rewards/comparison_report.png
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _float(val: str) -> float | None:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def generate_report(
    baseline_path: Path,
    trained_path: Path,
    log_path: Path,
    output_path: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend — safe on HPC nodes
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("matplotlib not installed. Run: pip install matplotlib")
        sys.exit(1)

    baseline = _load_json(baseline_path)
    trained  = _load_json(trained_path)
    log_rows = _load_csv(log_path)

    # ---- Figure layout: 2×2 grid ----------------------------------------
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("PR Review Agent — Training Report", fontsize=15, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax_acc   = fig.add_subplot(gs[0, 0])  # top-left:  accuracy comparison
    ax_ret   = fig.add_subplot(gs[0, 1])  # top-right: mean return comparison
    ax_lang  = fig.add_subplot(gs[1, 0])  # bot-left:  per-language accuracy
    ax_curve = fig.add_subplot(gs[1, 1])  # bot-right: training reward curve

    COLORS = {"random": "#9e9e9e", "heuristic": "#5c9bd6", "trained_slm": "#2ca02c"}
    LABELS = {"random": "Random", "heuristic": "Heuristic", "trained_slm": "Trained SLM"}

    # ---- Collect policy results ------------------------------------------
    policies: dict[str, dict] = {}
    if baseline:
        for name, data in baseline.items():
            policies[name] = data
    if trained:
        policies["trained_slm"] = trained

    policy_order = [p for p in ("random", "heuristic", "trained_slm") if p in policies]

    # ---- Panel 1: Accuracy bar chart ------------------------------------
    accs = [policies[p]["accuracy"] for p in policy_order]
    bars = ax_acc.bar(
        [LABELS.get(p, p) for p in policy_order],
        accs,
        color=[COLORS.get(p, "#cccccc") for p in policy_order],
        edgecolor="white",
        linewidth=1.2,
        width=0.5,
    )
    for bar, val in zip(bars, accs):
        ax_acc.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.1%}",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )
    ax_acc.set_title("Verdict Accuracy", fontweight="bold")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_ylim(0, 1.15)
    ax_acc.axhline(0.333, color="red", linestyle="--", linewidth=0.8, label="random-chance (3-class)")
    ax_acc.legend(fontsize=8)
    ax_acc.spines[["top", "right"]].set_visible(False)

    # ---- Panel 2: Mean return bar chart ---------------------------------
    rets = [policies[p]["mean_episode_return"] for p in policy_order]
    bars2 = ax_ret.bar(
        [LABELS.get(p, p) for p in policy_order],
        rets,
        color=[COLORS.get(p, "#cccccc") for p in policy_order],
        edgecolor="white",
        linewidth=1.2,
        width=0.5,
    )
    for bar, val in zip(bars2, rets):
        offset = 0.02 if val >= 0 else -0.06
        ax_ret.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            f"{val:+.3f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )
    ax_ret.set_title("Mean Episode Return", fontweight="bold")
    ax_ret.set_ylabel("Return")
    ax_ret.axhline(0, color="black", linewidth=0.6)
    ax_ret.spines[["top", "right"]].set_visible(False)

    # ---- Panel 3: Per-language accuracy (grouped bars) ------------------
    all_langs: list[str] = []
    for p in policy_order:
        by_lang = policies[p].get("by_language", {})
        for lang in by_lang:
            if lang not in all_langs:
                all_langs.append(lang)
    all_langs.sort()

    import numpy as np
    x = np.arange(len(all_langs))
    width = 0.25
    for i, p in enumerate(policy_order):
        by_lang = policies[p].get("by_language", {})
        vals = [by_lang.get(lang, {}).get("accuracy", 0.0) for lang in all_langs]
        ax_lang.bar(
            x + i * width - width,
            vals,
            width,
            label=LABELS.get(p, p),
            color=COLORS.get(p, "#cccccc"),
            edgecolor="white",
        )
    ax_lang.set_title("Accuracy by Language", fontweight="bold")
    ax_lang.set_ylabel("Accuracy")
    ax_lang.set_xticks(x)
    ax_lang.set_xticklabels(all_langs, rotation=30, ha="right", fontsize=9)
    ax_lang.set_ylim(0, 1.25)
    ax_lang.legend(fontsize=8)
    ax_lang.spines[["top", "right"]].set_visible(False)

    # ---- Panel 4: Training reward curve ---------------------------------
    if log_rows:
        steps = [int(r["step"]) for r in log_rows if r.get("step")]
        rewards = [_float(r.get("reward_mean", "")) for r in log_rows]
        losses  = [_float(r.get("loss", "")) for r in log_rows]

        reward_pairs = [(s, v) for s, v in zip(steps, rewards) if v is not None]
        loss_pairs   = [(s, v) for s, v in zip(steps, losses)  if v is not None]

        if reward_pairs:
            rs, rv = zip(*reward_pairs)
            ax_curve.plot(rs, rv, color="#2ca02c", linewidth=1.5, label="reward_mean")
        if loss_pairs:
            ax2 = ax_curve.twinx()
            ls, lv = zip(*loss_pairs)
            ax2.plot(ls, lv, color="#d62728", linewidth=1, linestyle="--", alpha=0.7, label="loss")
            ax2.set_ylabel("Loss", color="#d62728", fontsize=9)
            ax2.tick_params(axis="y", labelcolor="#d62728")
            ax2.legend(loc="upper right", fontsize=8)

        ax_curve.set_title("Training Curves (GRPO)", fontweight="bold")
        ax_curve.set_xlabel("Step")
        ax_curve.set_ylabel("Reward", color="#2ca02c")
        ax_curve.tick_params(axis="y", labelcolor="#2ca02c")
        ax_curve.legend(loc="upper left", fontsize=8)
        ax_curve.spines[["top"]].set_visible(False)
    else:
        ax_curve.text(
            0.5, 0.5,
            "training_log.csv not found\n(run training first)",
            ha="center", va="center", transform=ax_curve.transAxes,
            fontsize=10, color="gray",
        )
        ax_curve.set_title("Training Curves (GRPO)", fontweight="bold")
        ax_curve.spines[["top", "right"]].set_visible(False)

    # ---- Summary text at bottom -----------------------------------------
    lines = []
    for p in policy_order:
        d = policies[p]
        lines.append(
            f"{LABELS.get(p, p):<14}  acc={d['accuracy']:.1%}  "
            f"return={d['mean_episode_return']:+.3f}  "
            f"n={d['episodes']}"
        )
    fig.text(
        0.5, 0.01,
        "\n".join(lines),
        ha="center", va="bottom", fontsize=9,
        fontfamily="monospace",
        color="#444444",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Report saved → {output_path}")

    # Also print ASCII table to stdout
    print(f"\n{'Policy':<14}  {'Accuracy':>8}  {'Mean Return':>11}  {'Tasks':>5}")
    print("-" * 46)
    for p in policy_order:
        d = policies[p]
        print(f"{LABELS.get(p, p):<14}  {d['accuracy']:>8.1%}  {d['mean_episode_return']:>+11.3f}  {d['episodes']:>5}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate comparison report from reward artifacts.")
    parser.add_argument("--baseline", default=str(ROOT / "rewards" / "baseline_eval.json"))
    parser.add_argument("--trained",  default=str(ROOT / "rewards" / "trained_eval.json"))
    parser.add_argument("--log",      default=str(ROOT / "grpo_checkpoint" / "training_log.csv"))
    parser.add_argument("--output",   default=str(ROOT / "rewards" / "comparison_report.png"))
    args = parser.parse_args()

    generate_report(
        baseline_path=Path(args.baseline),
        trained_path=Path(args.trained),
        log_path=Path(args.log),
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
