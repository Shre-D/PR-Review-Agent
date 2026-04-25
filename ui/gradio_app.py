"""Gradio UI for the PR Review Agent — browse tasks, run reviews, try custom diffs."""

from __future__ import annotations

import difflib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import gradio as gr

from benchmarks.run_baselines import decide_final_verdict, heuristic_policy
from envs.pr_review_env.models import PRReviewAction, PRReviewObservation
from envs.pr_review_env.server.grader import (
    aggregate_tool_scores,
    get_scoring_logic,
    relevant_tools,
)
from envs.pr_review_env.server.pr_review_env import PRReviewEnv
from envs.pr_review_env.server.tasks import PRTask, load_tasks, task_review_config
from envs.pr_review_env.server.tools import (
    check_build_and_types,
    check_config,
    check_quality,
    check_security,
    check_tests,
)
from inference import run_ollama_step
from train.adaptive_router import prompt_route_context
from train.grpo_train import SYSTEM_PROMPT, _action_with_state_args, build_obs_prompt, parse_action, training_prompt


BLOG_DIR = ROOT / "blog"
BANKS = ["all", "seed", "comprehensive"]
LOADER_MODES = ["short", "empty", "full", "off"]

# Policy choices. Format:
#   "heuristic"                                  — deterministic baseline
#   "ollama:<tag>"                               — local Ollama LLM
#   "slm:<checkpoint_path>"                      — trained LoRA checkpoint on disk
#   "slm:<hf-user/repo>"                         — trained LoRA from HF Hub
DEFAULT_TRAINED_CKPT = os.getenv("PR_REVIEW_TRAINED_CKPT", "")
POLICIES = ["heuristic", "ollama:qwen2.5:7b"]
if DEFAULT_TRAINED_CKPT:
    POLICIES.append(f"slm:{DEFAULT_TRAINED_CKPT}")
DEFAULT_POLICY = os.getenv("PR_REVIEW_UI_POLICY", "heuristic")
MAX_AGENT_STEPS = 8

# Per-1k-token cost estimates for the cost-comparison panel (USD, Apr 2026).
# Local Qwen3-1.7B on CPU ~ amortised electricity cost; treated as ~$0.0001/review.
COST_PER_1K_TOKENS = {
    "slm-qwen3-1p7b-local": 0.00001,
    "gpt-4o-mini":          0.00015,
    "claude-haiku-4-5":     0.0008,
    "claude-sonnet-4-5":    0.003,
    "gpt-4o":               0.005,
}

TOOL_FUNCS = {
    "check_security": check_security,
    "check_quality": check_quality,
    "check_build_and_types": check_build_and_types,
    "check_tests": check_tests,
    "check_config": check_config,
}

EXAMPLE_BEFORE = """from flask import request

def search_users(conn):
    query = request.args.get("q", "")
    return conn.execute("SELECT * FROM users WHERE name = ?", (query,)).fetchall()
"""
EXAMPLE_AFTER = """from flask import request

def search_users(conn):
    query = request.args.get("q", "")
    sql = f"SELECT * FROM users WHERE name LIKE '%{query}%'"
    return conn.execute(sql).fetchall()
"""

VERDICT_LABEL = {
    "approve": "✅  APPROVE",
    "request_changes": "🔴  REQUEST CHANGES",
    "comment": "💬  COMMENT ONLY",
}

DIFFICULTY_LABEL = {"easy": "🟢 Easy", "medium": "🟡 Medium", "hard": "🔴 Hard"}
AUTHOR_LABEL = {
    "junior": "Junior",
    "mid": "Mid-level",
    "senior": "Senior",
    "lead": "Lead",
    "non_tech": "Non-technical",
    "unknown": "Unknown",
}


def _json(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True)


def _task(task_bank: str, task_id: str) -> PRTask:
    return next(task for task in load_tasks(task_bank) if task.task_id == task_id)


def task_choices(task_bank: str) -> list[str]:
    return [task.task_id for task in load_tasks(task_bank)]


def blog_choices() -> list[str]:
    return [path.name for path in sorted(BLOG_DIR.glob("*.md"))]


def read_blog(name: str) -> str:
    if not name:
        return ""
    path = BLOG_DIR / name
    if not path.exists():
        return "Blog file not found."
    return path.read_text(encoding="utf-8")


TASK_TABLE_HEADERS = ["ID", "Expected Verdict", "Language", "Difficulty", "Risk Areas", "File Types", "Author Level"]


def task_table(task_bank: str) -> list[list[Any]]:
    rows = []
    for task in load_tasks(task_bank):
        rows.append([
            task.task_id,
            task.expected_verdict,
            task.primary_language,
            task.difficulty,
            ", ".join(task.risk_domains),
            ", ".join(task.changed_file_types),
            task.author_level,
        ])
    return rows


def task_stats_md(task_bank: str) -> str:
    tasks = load_tasks(task_bank)
    by_verdict = Counter(task.expected_verdict for task in tasks)
    by_language = Counter(task.primary_language for task in tasks)
    by_difficulty = Counter(task.difficulty for task in tasks)

    lines = [f"**{len(tasks)} tasks total**\n"]
    lines.append("**Verdict breakdown**")
    for k, v in sorted(by_verdict.items()):
        lines.append(f"- {k}: {v}")
    lines.append("\n**By language**")
    for k, v in sorted(by_language.items(), key=lambda x: -x[1]):
        lines.append(f"- {k}: {v}")
    lines.append("\n**By difficulty**")
    for k, v in sorted(by_difficulty.items()):
        lines.append(f"- {DIFFICULTY_LABEL.get(k, k)}: {v}")
    return "\n".join(lines)


def _task_card(task: PRTask, obs_risk: str | None, obs_critical: list[str] | None, tools_selected: list[str]) -> str:
    verdict = VERDICT_LABEL.get(task.expected_verdict, task.expected_verdict)
    difficulty = DIFFICULTY_LABEL.get(task.difficulty, task.difficulty)
    author = AUTHOR_LABEL.get(task.author_level, task.author_level)
    risks = ", ".join(task.risk_domains) if task.risk_domains else "none"
    file_types = ", ".join(task.changed_file_types)
    tools = ", ".join(f"`{t}`" for t in tools_selected) if tools_selected else "none"
    critical = ", ".join(obs_critical) if obs_critical else "none"

    lines = [
        f"### {task.task_id}",
        f"**Expected verdict:** {verdict}",
        f"**Language:** {task.primary_language}  |  **Repo kind:** {task.repo_kind}",
        f"**Difficulty:** {difficulty}  |  **Author:** {author}",
        f"**Risk areas:** {risks}",
        f"**File types:** {file_types}",
        f"**Review goal:** {task.review_goal}",
    ]
    if task.ownership_hint:
        lines.append(f"**Ownership hint:** {task.ownership_hint}")
    lines += [
        "",
        f"**Risk level (estimated):** {obs_risk or 'N/A'}",
        f"**Critical paths touched:** {critical}",
        f"**Tools the policy will run:** {tools}",
    ]
    return "\n".join(lines)


def scoring_guide_md() -> str:
    logic = get_scoring_logic()
    tp = logic["tool_penalties"]
    return f"""
### 📊 Understanding the Scores

| Metric | What it measures | Calculation |
| :--- | :--- | :--- |
| **Tool Score** | Severity of findings in a single tool call. | Starts at `1.0`. Subtracts penalty per finding (e.g., security=`-{tp['check_security']}`, tests=`-{tp['check_tests']}`). |
| **Aggregate Score** | Overall "health" of the PR across all tools. | Weighted average of Tool Scores. Relevant tools are weighted **{logic['relevant_weight_multiplier']}x**. |
| **Reward** | The RL signal for the agent's performance. | `Base(0.05) + Relevance Bonus(0.08) - Efficiency Penalties`. Terminal correct = `~{logic['terminal_correct_base']}`. |

**Why did the score change?**
Tool scores are **not cumulative**. If `check_security` finds an issue (0.1) but `check_quality` is clean (1.0), the table shows both. The **Aggregate Score** at the top combines them into a single verdict-aligned metric.
"""


def task_overview(task_bank: str, task_id: str, loader_mode: str) -> tuple[str, str, str, str, str, str]:
    task = _task(task_bank, task_id)
    config = task_review_config(task, mode=loader_mode)
    env = PRReviewEnv(seed=7, task_path=task_bank, review_config=config)
    obs = env.reset(task_id=task.task_id)

    tools_selected = sorted(relevant_tools(task, env._review_config))  # noqa: SLF001
    plan = [action.tool_name for action in heuristic_policy(obs)]
    route = prompt_route_context(obs, config)
    prompt = training_prompt(obs, config)
    card = _task_card(task, obs.estimated_risk_level, obs.critical_paths_touched, tools_selected)
    return card, task.diff_str, _json(config), route, "\n".join(plan), prompt


def _verdict_card(expected: str, predicted: str, episode_return: float, tools_called: list[str], agg_score: Any) -> str:
    correct = predicted == expected
    status = "✅ Correct" if correct else "❌ Incorrect"
    pred_label = VERDICT_LABEL.get(predicted, predicted)
    exp_label = VERDICT_LABEL.get(expected, expected)
    tools = ", ".join(f"`{t}`" for t in tools_called) if tools_called else "none"
    score_str = f"{agg_score:.3f}" if isinstance(agg_score, float) else str(agg_score or "N/A")
    
    explanation = (
        "**Aggregate Score** is the weighted average of all Tool Scores. "
        "Closer to 1.0 means the PR is 'cleaner'; closer to 0.0 means critical issues were found."
    )
    
    return "\n".join([
        f"## Result: {status}",
        f"**Predicted Verdict:** {pred_label}",
        f"**Expected Verdict:** {exp_label}",
        f"**Total Episode Reward:** `{episode_return}`",
        f"**Final Aggregate Score:** `{score_str}`",
        f"**Tools used for evidence:** {tools}",
        "",
        f"> {explanation}"
    ])


def _trace_markdown(trace: list[dict[str, Any]]) -> str:
    sections = []
    for index, item in enumerate(trace, start=1):
        action = item["action"]["tool_name"]
        result = item.get("result") or {}
        findings = result.get("findings") or []
        score = result.get("score", "")
        mode = result.get("analysis_mode", "")
        reward = item.get("reward")

        header_parts = [f"### {index}. `{action}`"]
        if score != "":
            header_parts.append(f"Severity Score: `{score}`")
        if reward is not None:
            header_parts.append(f"Reward: `{reward}`")

        if action in {"submit_review", "escalate"}:
            verdict = result.get("verdict") or item["action"].get("arguments", {}).get("verdict", "")
            confidence = result.get("confidence") or item["action"].get("arguments", {}).get("confidence")
            reasoning = result.get("reasoning") or item["action"].get("arguments", {}).get("reasoning", "")
            
            lines = [
                f"## 🏁 Final Submission",
                f"- **Verdict:** **{verdict.upper()}**",
                f"- **Confidence:** `{confidence}`" if confidence is not None else "",
                f"### Reasoning:",
                f"> {reasoning}" if reasoning else "_No reasoning provided._"
            ]
            rendered = "\n".join(filter(None, lines))
        elif findings:
            rendered = "\n".join(f"- {f}" for f in findings)
        else:
            rendered = "_No findings (Tool reported clean result)_"

        sections.append(" | ".join(header_parts) + "\n\n" + rendered)
    return "\n\n---\n\n".join(sections)


def _trace_findings_cell(item: dict[str, Any]) -> str:
    action = item["action"]["tool_name"]
    result = item.get("result") or {}
    if action in {"submit_review", "escalate"}:
        verdict = result.get("verdict") or item["action"].get("arguments", {}).get("verdict", "")
        confidence = result.get("confidence") or item["action"].get("arguments", {}).get("confidence")
        parts = [f"verdict={verdict}"]
        if confidence is not None:
            parts.append(f"confidence={confidence}")
        return "; ".join(parts)
    return "; ".join(result.get("findings") or [])


def _ollama_action(model_name: str, obs: PRReviewObservation, config: dict | None) -> PRReviewAction:
    """Single agent step from Ollama. Falls back to a safe submit_review on failure."""
    try:
        action = run_ollama_step(model_name, obs)
        return _action_with_state_args(action, obs)
    except Exception as exc:  # noqa: BLE001 — surface via UI, do not crash
        return PRReviewAction(
            tool_name="submit_review",
            arguments={
                "verdict": "escalate",
                "confidence": 0.0,
                "reasoning": f"Ollama error: {type(exc).__name__}: {exc}",
            },
        )


def _ollama_model(policy: str) -> str:
    return policy.split("ollama:", 1)[1] if policy.startswith("ollama:") else policy


def _run_episode(
    env: PRReviewEnv,
    obs: PRReviewObservation,
    policy: str,
    config: dict | None,
) -> tuple[list[dict[str, Any]], float, PRReviewObservation]:
    trace: list[dict[str, Any]] = []
    total = 0.0

    if policy == "heuristic":
        for action in heuristic_policy(obs):
            obs = env.step(action)
            total += obs.reward or 0.0
            trace.append({"action": action.model_dump(), "reward": obs.reward, "result": obs.last_tool_result})
        final = decide_final_verdict(obs)
        obs = env.step(final)
        total += obs.reward or 0.0
        trace.append({"action": final.model_dump(), "reward": obs.reward, "result": obs.last_tool_result})
        return trace, total, obs

    # Ollama-driven multi-step agent loop
    model_name = _ollama_model(policy)
    for _ in range(MAX_AGENT_STEPS):
        action = _ollama_action(model_name, obs, config)
        obs = env.step(action)
        total += obs.reward or 0.0
        trace.append({"action": action.model_dump(), "reward": obs.reward, "result": obs.last_tool_result})
        if obs.done:
            break

    if not obs.done:
        final = decide_final_verdict(obs)
        obs = env.step(final)
        total += obs.reward or 0.0
        trace.append({"action": final.model_dump(), "reward": obs.reward, "result": obs.last_tool_result})

    return trace, total, obs


def run_task_review(
    task_bank: str,
    task_id: str,
    loader_mode: str,
    policy: str = DEFAULT_POLICY,
) -> tuple[str, list[list[Any]], str, str]:
    task = _task(task_bank, task_id)
    config = task_review_config(task, mode=loader_mode)
    env = PRReviewEnv(seed=7, task_path=task_bank, review_config=config)
    obs = env.reset(task_id=task.task_id)

    trace, total, obs = _run_episode(env, obs, policy, config)

    final_action = trace[-1]["action"]
    verdict = final_action.get("arguments", {}).get("verdict") or final_action.get("tool_name", "")
    agg = obs.metadata.get("aggregate_score") if obs.metadata else None
    verdict_card = _verdict_card(task.expected_verdict, verdict, round(total, 4), obs.tools_called, agg)

    rows = [
        [
            idx,
            item["action"]["tool_name"],
            item.get("reward"),
            (item.get("result") or {}).get("score"),
            (item.get("result") or {}).get("analysis_mode"),
            _trace_findings_cell(item),
        ]
        for idx, item in enumerate(trace, start=1)
    ]
    return verdict_card, rows, _trace_markdown(trace), _json(trace)


def _detect_file_types(path: str) -> list[str]:
    lower = path.lower()
    name = Path(lower).name
    if lower.endswith(".py"):
        return ["python"]
    if lower.endswith((".ts", ".tsx")):
        return ["typescript"]
    if lower.endswith((".js", ".jsx")):
        return ["javascript"]
    if lower.endswith(".java"):
        return ["java"]
    if lower.endswith(".go"):
        return ["go"]
    if lower.endswith(".rs"):
        return ["rust"]
    if lower.endswith((".yml", ".yaml")):
        return ["yaml", "github_actions"] if ".github/workflows/" in lower else ["yaml"]
    if name == "dockerfile":
        return ["dockerfile"]
    if name == ".gitignore":
        return ["gitignore"]
    if name in {"package.json", "pom.xml", "cargo.toml", "go.mod"}:
        return ["build_manifest"]
    return ["text"]


def make_diff(path: str, before: str, after: str) -> str:
    path = path.strip() or "app.py"
    diff_lines = difflib.unified_diff(
        before.splitlines(),
        after.splitlines(),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
        lineterm="",
    )
    rendered = "\n".join(diff_lines)
    if not rendered:
        return f"diff --git a/{path} b/{path}\n# no changes"
    return f"diff --git a/{path} b/{path}\n{rendered}\n"


def custom_context(
    path: str,
    before: str,
    after: str,
    pr_description: str,
    risk_domains_csv: str,
    loader_mode: str,
) -> tuple[str, str, str, str, str]:
    diff = make_diff(path, before, after)
    file_types = _detect_file_types(path)
    primary = next(
        (t for t in file_types if t in {"python", "typescript", "javascript", "java", "go", "rust"}),
        file_types[0],
    )
    risk_domains = [t.strip() for t in risk_domains_csv.split(",") if t.strip()]
    task = PRTask(
        task_id="custom_ui_example",
        diff_str=diff,
        pr_description=pr_description or "Custom before/after diff",
        primary_language=primary,
        changed_file_types=file_types,
        repo_kind="custom",
        expected_verdict="request_changes",
        risk_domains=risk_domains,
        difficulty="medium",
        review_goal="Interactive UI review of a custom diff.",
    )
    config = task_review_config(task, mode=loader_mode)
    obs = PRReviewObservation(
        diff_str=diff,
        pr_description=task.pr_description,
        primary_language=primary,
        changed_file_types=file_types,
        repo_kind=task.repo_kind,
        available_tools=list(TOOL_FUNCS),
        task_id=task.task_id,
    )
    tools_selected = sorted(relevant_tools(task))
    detected_md = "\n".join([
        f"**Language detected:** {primary}",
        f"**File types:** {', '.join(file_types)}",
        f"**Risk areas:** {', '.join(risk_domains) if risk_domains else 'none'}",
        f"**Tools the policy will run:** {', '.join(f'`{t}`' for t in tools_selected) if tools_selected else 'none'}",
    ])
    return diff, detected_md, _json(config), prompt_route_context(obs, config), training_prompt(obs, config)


def run_custom_review(
    path: str,
    before: str,
    after: str,
    pr_description: str,
    risk_domains_csv: str,
    loader_mode: str,
    policy: str = DEFAULT_POLICY,
) -> tuple[str, list[list[Any]], str, str]:
    diff, _detected_md, config_json, _route, _prompt = custom_context(
        path, before, after, pr_description, risk_domains_csv, loader_mode,
    )
    config = json.loads(config_json) if config_json != "null" else None
    file_types = _detect_file_types(path)
    primary = next(
        (t for t in file_types if t in {"python", "typescript", "javascript", "java", "go", "rust"}),
        file_types[0],
    )
    risk_domains = [t.strip() for t in risk_domains_csv.split(",") if t.strip()]
    obs = PRReviewObservation(
        diff_str=diff,
        pr_description=pr_description or "Custom before/after diff",
        primary_language=primary,
        changed_file_types=file_types,
        repo_kind="custom",
        available_tools=list(TOOL_FUNCS),
        task_id="custom_ui_example",
    )

    if policy.startswith("ollama:"):
        actions: list[PRReviewAction] = []
        model_name = _ollama_model(policy)
        running_obs = obs
        for _ in range(MAX_AGENT_STEPS):
            action = _ollama_action(model_name, running_obs, config)
            if action.tool_name in {"submit_review", "escalate"}:
                actions.append(action)
                break
            actions.append(action)
            tool_name = action.tool_name
            if tool_name not in TOOL_FUNCS:
                break
            result = TOOL_FUNCS[tool_name](diff_str=diff, task_id="", review_config=config)
            running_obs = PRReviewObservation(
                diff_str=diff,
                pr_description=obs.pr_description,
                primary_language=obs.primary_language,
                changed_file_types=obs.changed_file_types,
                repo_kind=obs.repo_kind,
                available_tools=list(TOOL_FUNCS),
                tool_results={**running_obs.tool_results, tool_name: result},
                tools_called=[*running_obs.tools_called, tool_name],
                task_id=obs.task_id,
            )
    else:
        actions = heuristic_policy(obs)

    tool_results: dict[str, dict[str, Any]] = {}
    trace: list[dict[str, Any]] = []

    for action in actions:
        if action.tool_name in {"submit_review", "escalate"}:
            continue
        if action.tool_name not in TOOL_FUNCS:
            continue
        fn = TOOL_FUNCS[action.tool_name]
        result = fn(diff_str=diff, task_id="", review_config=config)
        tool_results[action.tool_name] = result
        trace.append({"action": action.model_dump(), "reward": None, "result": result})

    scored_obs = PRReviewObservation(
        diff_str=diff,
        pr_description=obs.pr_description,
        primary_language=obs.primary_language,
        changed_file_types=obs.changed_file_types,
        repo_kind=obs.repo_kind,
        available_tools=list(TOOL_FUNCS),
        tool_results=tool_results,
        task_id=obs.task_id,
    )
    submitted = next((a for a in actions if a.tool_name in {"submit_review", "escalate"}), None)
    if submitted is None:
        final = decide_final_verdict(scored_obs)
    else:
        final = submitted
    trace.append({"action": final.model_dump(), "reward": None, "result": final.arguments})

    verdict = final.arguments.get("verdict", final.tool_name)
    reasoning = final.arguments.get("reasoning", "")
    agg = aggregate_tool_scores(tool_results)
    tools_run = [a.tool_name for a in actions if a.tool_name in TOOL_FUNCS]

    verdict_card = "\n".join([
        f"## {VERDICT_LABEL.get(verdict, verdict)}",
        f"**Policy:** `{policy}` — risk areas: {', '.join(risk_domains) or 'none'}",
        f"**Aggregate score:** `{agg:.3f}` (no ground-truth reward for custom examples)",
        f"**Tools run:** {', '.join(f'`{t}`' for t in tools_run) or 'none'}",
        "",
        f"**Reasoning:** {reasoning}",
    ])

    rows = [
        [
            idx,
            item["action"]["tool_name"],
            (item.get("result") or {}).get("score"),
            (item.get("result") or {}).get("analysis_mode"),
            _trace_findings_cell(item),
        ]
        for idx, item in enumerate(trace, start=1)
    ]
    return verdict_card, rows, _trace_markdown(trace), _json(trace)


def refresh_bank(task_bank: str, loader_mode: str):
    choices = task_choices(task_bank)
    first = choices[0]
    overview = task_overview(task_bank, first, loader_mode)
    return (
        gr.update(choices=choices, value=first),
        task_table(task_bank),
        task_stats_md(task_bank),
        *overview,
    )


def build_app() -> gr.Blocks:
    with gr.Blocks(title="PR Review Agent") as demo:
        gr.Markdown("# PR Review Agent\nBrowse benchmark tasks, inspect how the policy reasons about them, and try it on your own code changes.")

        with gr.Tabs():
            with gr.Tab("Task Explorer"):
                gr.Markdown("Select a task from the benchmark to see how the agent analyzes it, which tools it picks, and how it performs.")

                with gr.Row():
                    bank = gr.Dropdown(
                        choices=BANKS,
                        value="all",
                        label="Dataset",
                        info="Which subset of tasks to show",
                        scale=1,
                    )
                    loader = gr.Dropdown(
                        choices=LOADER_MODES,
                        value="short",
                        label="Context loading",
                        info="How much repo context is injected",
                        scale=1,
                    )
                    policy = gr.Dropdown(
                        choices=POLICIES,
                        value=DEFAULT_POLICY,
                        label="Policy",
                        info="heuristic = deterministic baseline; ollama:qwen2.5:7b = local LLM",
                        scale=1,
                    )
                    task = gr.Dropdown(
                        choices=task_choices("all"),
                        value=task_choices("all")[0],
                        label="Task",
                        scale=3,
                    )

                with gr.Row():
                    with gr.Column(scale=2):
                        task_card = gr.Markdown(label="Task overview")
                    with gr.Column(scale=1):
                        stats = gr.Markdown(value=task_stats_md("all"), label="Dataset stats")

                table = gr.Dataframe(
                    value=task_table("all"),
                    headers=TASK_TABLE_HEADERS,
                    label="All tasks in dataset",
                    interactive=False,
                    wrap=True,
                )

                diff = gr.Code(label="PR diff", language=None, lines=20)

                with gr.Accordion("Advanced: loader config & model prompt", open=False):
                    with gr.Row():
                        route = gr.Textbox(label="Router decision", lines=6, info="What the adaptive router decides about this task")
                        plan = gr.Textbox(label="Tool execution plan", lines=6, info="Which tools the heuristic policy will run")
                    loader_config = gr.Code(label="Loader config (passed to tools)", language="json", lines=12)
                    prompt = gr.Code(label="Full prompt sent to the model", language="markdown", lines=20)

                run_btn = gr.Button("Run review (policy above)", variant="primary", size="lg")

                with gr.Accordion("📚 Scoring & Reward Guide", open=False):
                    gr.Markdown(scoring_guide_md())

                verdict_card = gr.Markdown(label="Review result")

                with gr.Row():
                    trace_table = gr.Dataframe(
                        headers=["Step", "Tool", "Reward", "Tool Score (Severity)", "Mode", "Findings"],
                        label="Tool results",
                        interactive=False,
                        wrap=True,
                    )

                findings_md = gr.Markdown(label="Findings per tool")

                with gr.Accordion("Raw trace JSON", open=False):
                    trace_json = gr.Code(label="Full trace", language="json", lines=18)

                bank.change(
                    refresh_bank,
                    inputs=[bank, loader],
                    outputs=[task, table, stats, task_card, diff, loader_config, route, plan, prompt],
                )
                for trigger in (task.change, loader.change):
                    trigger(
                        task_overview,
                        inputs=[bank, task, loader],
                        outputs=[task_card, diff, loader_config, route, plan, prompt],
                    )
                run_btn.click(
                    run_task_review,
                    inputs=[bank, task, loader, policy],
                    outputs=[verdict_card, trace_table, findings_md, trace_json],
                )

            with gr.Tab("Try Your Own Change"):
                gr.Markdown(
                    "Paste code before and after a change — the agent will diff them and run its review tools. "
                    "No ground-truth verdict exists for custom examples, but you can see what the policy decides and why."
                )

                with gr.Row():
                    custom_path = gr.Textbox(
                        value="app/search.py",
                        label="File path",
                        info="Used to detect the language and file type",
                        scale=3,
                    )
                    custom_loader = gr.Dropdown(
                        choices=LOADER_MODES,
                        value="short",
                        label="Context loading",
                        scale=1,
                    )
                    custom_policy = gr.Dropdown(
                        choices=POLICIES,
                        value=DEFAULT_POLICY,
                        label="Policy",
                        scale=1,
                    )

                with gr.Row():
                    custom_pr = gr.Textbox(
                        value="Adds a Flask search endpoint for customer records.",
                        label="PR description",
                        lines=2,
                        scale=3,
                    )
                    custom_risks = gr.Textbox(
                        value="security",
                        label="Risk areas (comma-separated)",
                        info="e.g. security, performance, correctness",
                        lines=2,
                        scale=1,
                    )

                with gr.Row():
                    before = gr.Code(value=EXAMPLE_BEFORE, label="Before", language="python", lines=18)
                    after = gr.Code(value=EXAMPLE_AFTER, label="After", language="python", lines=18)

                diff_btn = gr.Button("Generate Diff", variant="secondary", size="sm")

                detected_md = gr.Markdown(label="Auto-detected metadata")

                with gr.Accordion("Generated diff & advanced options", open=False):
                    custom_diff = gr.Code(label="Unified diff (auto-generated)", language=None, lines=16)
                    with gr.Row():
                        custom_route = gr.Textbox(label="Router decision", lines=5)
                    custom_config = gr.Code(label="Loader config", language="json", lines=12)
                    custom_prompt = gr.Code(label="Model prompt", language="markdown", lines=20)

                analyze_btn = gr.Button("Analyze this change", variant="primary", size="lg")

                with gr.Accordion("📚 Scoring & Reward Guide", open=False):
                    gr.Markdown(scoring_guide_md())

                custom_verdict = gr.Markdown(label="Review result")

                with gr.Row():
                    custom_trace_table = gr.Dataframe(
                        headers=["Step", "Tool", "Tool Score (Severity)", "Mode", "Findings"],
                        label="Tool results",
                        interactive=False,
                        wrap=True,
                    )

                custom_findings = gr.Markdown(label="Findings per tool")

                with gr.Accordion("Raw trace JSON", open=False):
                    custom_trace = gr.Code(label="Full trace", language="json", lines=18)

                ctx_inputs = [custom_path, before, after, custom_pr, custom_risks, custom_loader]
                ctx_outputs = [custom_diff, detected_md, custom_config, custom_route, custom_prompt]
                diff_btn.click(custom_context, inputs=ctx_inputs, outputs=ctx_outputs)
                for trigger in (
                    custom_path.change,
                    custom_pr.change,
                    custom_risks.change,
                    custom_loader.change,
                ):
                    trigger(custom_context, inputs=ctx_inputs, outputs=ctx_outputs)
                analyze_btn.click(
                    run_custom_review,
                    inputs=[custom_path, before, after, custom_pr, custom_risks, custom_loader, custom_policy],
                    outputs=[custom_verdict, custom_trace_table, custom_findings, custom_trace],
                )

            with gr.Tab("Blog"):
                blogs = blog_choices()
                blog_select = gr.Dropdown(blogs, value=blogs[0] if blogs else None, label="Post")
                blog_body = gr.Markdown(read_blog(blogs[0]) if blogs else "No blog posts found.")
                blog_select.change(read_blog, inputs=blog_select, outputs=blog_body)

        demo.load(
            task_overview,
            inputs=[bank, task, loader],
            outputs=[task_card, diff, loader_config, route, plan, prompt],
        )

    demo.queue(default_concurrency_limit=4)
    return demo


if __name__ == "__main__":
    build_app().launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        theme=gr.themes.Soft(),
    )
