"""Post-training context loader benchmark.

Compares the quality of review_config.json produced by:
  - Structural parsing only (no model)
  - Qwen3-1.7B SLM extraction

Then runs the trained router against a selected task bank and reports accuracy
and mean return per config.

This is a MANUAL benchmark — run after training is complete.
Not called during training. No LLM calls in the training path.

Usage:
    # Compare structural vs SLM extraction (no API key needed):
    python benchmarks/loader_benchmark.py \
        --docs-dir docs/ \
        --checkpoint grpo_checkpoint/ \
        --env-url http://localhost:8000 \
        --task-bank all

    # Optional: compare against another local model.
    python benchmarks/loader_benchmark.py \
        --docs-dir docs/ \
        --checkpoint grpo_checkpoint/ \
        --env-url http://localhost:8000 \
        --compare-slm Qwen/Qwen3-1.7B \
        --task-bank comprehensive \
        --output rewards/loader_benchmark.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.pr_review_env.client.pr_review_env_client import PRReviewEnvClient
from envs.pr_review_env.models import PRReviewAction
from envs.pr_review_env.server.context_loader import extract_structural_config
from envs.pr_review_env.server.tasks import PRTask, load_tasks
from train.grpo_train import parse_action


def extract_slm_config(docs_dir: Path, model_id: str) -> dict:
    """Use a local SLM to extract config from all docs/*.md files.

    Falls back to structural config if model not available.
    This is intentionally lazy-imported so the benchmark file stays importable
    without trl/transformers installed.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except ImportError:
        print(f"[loader_benchmark] transformers not installed, falling back to structural for {model_id}")
        return {**extract_structural_config(docs_dir), "extraction_method": f"fallback_structural (wanted {model_id})"}

    print(f"[loader_benchmark] Loading {model_id} for extraction...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )

    # Collect all non-review-tool markdown docs
    doc_contents = []
    for md_file in sorted(docs_dir.glob("*.md")):
        if md_file.name == "review-tool.md":
            continue
        content = md_file.read_text(encoding="utf-8")[:3000]  # cap per doc
        doc_contents.append(f"=== {md_file.name} ===\n{content}")

    combined_docs = "\n\n".join(doc_contents)[:8000]  # total cap

    extraction_prompt = f"""\
Extract structured code review configuration from these organisation documents.
Output ONLY valid JSON with this schema:
{{
  "custom_rules": [{{"domain": "security"|"quality"|"config"|"build"|"tests", "pattern": "regex", "severity": "critical"|"warning"|"info", "message": "human-readable finding"}}],
  "architecture_summary": "2 sentence summary of the tech stack",
  "domain_priorities": {{"security": 1.0, "quality": 1.0, "config": 1.0, "build": 1.0, "tests": 1.0}},
  "critical_paths": ["list of file path prefixes"]
}}

Documents:
{combined_docs}
"""

    messages = [
        {"role": "system", "content": "You are a configuration extraction assistant. Output only JSON."},
        {"role": "user", "content": extraction_prompt},
    ]

    t0 = time.time()
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(next(model.parameters()).device)

    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=512, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)

    raw = tokenizer.decode(out[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()
    elapsed = round(time.time() - t0, 1)
    print(f"[loader_benchmark] Extraction took {elapsed}s")

    # Parse JSON from output
    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        extracted = json.loads(raw[start:end])
    except Exception as e:
        print(f"[loader_benchmark] JSON parse failed: {e}. Using structural fallback.")
        extracted = {}

    # Merge structural (high-confidence) over LLM extraction
    structural = extract_structural_config(docs_dir)
    merged = {
        **extracted,
        "tool_weights": structural.get("tool_weights", {}),
        "reject_threshold": structural.get("reject_threshold", 0.30),
        "request_changes_threshold": structural.get("request_changes_threshold", 0.65),
        "extraction_method": model_id,
        "extraction_time_s": elapsed,
        "raw_output": raw[:500],  # store first 500 chars for inspection
    }
    return merged


# ---------------------------------------------------------------------------
# Episode runner using a trained checkpoint
# ---------------------------------------------------------------------------

async def run_episode_with_config(
    env_client: PRReviewEnvClient,
    task: PRTask,
    config: dict,
    checkpoint_path: str,
) -> dict:
    """Run one episode against the env, injecting config context into the prompt.

    If checkpoint_path is empty, falls back to heuristic policy for smoke testing.
    """
    step_result = await env_client.reset(task_id=task.task_id)
    obs = step_result.observation

    if not checkpoint_path:
        # Heuristic fallback (no model needed) for smoke testing
        from benchmarks.run_baselines import heuristic_policy, decide_final_verdict
        total_reward = 0.0
        for action in heuristic_policy(obs):
            step_result = await env_client.step(action)
            total_reward += step_result.reward or 0.0
            obs = step_result.observation
        final = decide_final_verdict(obs)
        step_result = await env_client.step(final)
        total_reward += step_result.reward or 0.0
        return {
            "task_id": task.task_id,
            "verdict": final.arguments.get("verdict"),
            "expected": task.expected_verdict,
            "correct": final.arguments.get("verdict") == task.expected_verdict,
            "episode_return": round(total_reward, 3),
            "policy": "heuristic_fallback",
        }

    # Trained model inference
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise RuntimeError("transformers + peft required for model inference")

    # Lazy model loading (cached between calls via module-level dict)
    if not hasattr(run_episode_with_config, "_model_cache"):
        run_episode_with_config._model_cache = {}

    if checkpoint_path not in run_episode_with_config._model_cache:
        print(f"[loader_benchmark] Loading checkpoint {checkpoint_path}...")
        base = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-1.7B", torch_dtype=torch.bfloat16, device_map="auto"
        )
        model = PeftModel.from_pretrained(base, checkpoint_path)
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
        run_episode_with_config._model_cache[checkpoint_path] = (model, tokenizer)

    model, tokenizer = run_episode_with_config._model_cache[checkpoint_path]

    # Build config-aware system prompt
    arch = config.get("architecture_summary", "")
    crit = ", ".join(config.get("critical_paths", []))
    system = (
        "You are a code reviewer. Output JSON tool calls only.\n"
        "Tools: check_security, check_quality, check_build_and_types, check_tests, check_config, submit_review, escalate.\n"
        'submit_review: {"tool_name": "submit_review", "arguments": {"verdict": "<approve|request_changes|reject>", "confidence": 0.9, "reasoning": "..."}}\n'
    )
    if arch:
        system += f"Architecture: {arch}\n"
    if crit:
        system += f"Critical paths: {crit}\n"

    messages = [{"role": "system", "content": system}]
    obs_text = (
        f"Language: {obs.primary_language}\nFiles: {', '.join(obs.changed_file_types)}\n"
        f"Author: {getattr(task, 'author_level', 'mid')}\n\n"
        f"PR: {obs.pr_description}\n\nDiff:\n{obs.diff_str}"
    )
    messages.append({"role": "user", "content": obs_text})

    total_reward = 0.0
    last_verdict = "approve"

    for _ in range(8):
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(next(model.parameters()).device)

        with torch.no_grad():
            out = model.generate(input_ids, max_new_tokens=128, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(out[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()

        action = parse_action(response)
        tool_name = action.tool_name
        arguments = action.arguments
        step_result = await env_client.step(action)
        total_reward += step_result.reward or 0.0
        obs = step_result.observation

        messages.append({"role": "assistant", "content": response})

        if tool_name == "submit_review":
            last_verdict = arguments.get("verdict", "approve")
            break
        if obs.done:
            break

        messages.append({"role": "user", "content": f"Review so far: {obs.review_history}\nContinue."})

    return {
        "task_id": task.task_id,
        "verdict": last_verdict,
        "expected": task.expected_verdict,
        "correct": last_verdict == task.expected_verdict,
        "episode_return": round(total_reward, 3),
        "policy": f"grpo+{checkpoint_path}",
    }


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

async def run_benchmark(
    docs_dir: Path,
    checkpoint: str,
    env_url: str,
    tasks_file: str,
    compare_slm: str,
    compare_llm: str,
    output: str,
) -> None:
    # Load tasks
    from envs.pr_review_env.server.tasks import load_tasks
    tasks = load_tasks(tasks_file)
    print(f"Using {len(tasks)} tasks from {tasks_file}")

    # Build configs to compare
    configs_to_test: list[tuple[str, dict]] = []

    print("\n=== Extracting configs ===")
    t0 = time.time()
    configs_to_test.append(("structural", extract_structural_config(docs_dir)))
    print(f"  structural: {time.time()-t0:.1f}s")

    if compare_slm:
        t0 = time.time()
        configs_to_test.append((f"slm:{compare_slm}", extract_slm_config(docs_dir, compare_slm)))
        print(f"  slm ({compare_slm}): {time.time()-t0:.1f}s")

    if compare_llm:
        t0 = time.time()
        configs_to_test.append((f"llm:{compare_llm}", extract_slm_config(docs_dir, compare_llm)))
        print(f"  llm ({compare_llm}): {time.time()-t0:.1f}s")

    # Run episodes for each config
    all_results: dict[str, list] = {}
    async with PRReviewEnvClient(base_url=env_url) as client:
        for config_name, config in configs_to_test:
            print(f"\n=== Running {len(tasks)} episodes with config: {config_name} ===")
            results = []
            for task in tasks:
                result = await run_episode_with_config(client, task, config, checkpoint)
                results.append(result)
                mark = "✓" if result["correct"] else "✗"
                print(f"  {mark} {task.task_id}: {result['verdict']} (return={result['episode_return']:+.2f})")
            all_results[config_name] = results

    # Summarise
    print("\n=== Results ===")
    print(f"{'Config':<30}  {'Accuracy':>8}  {'Mean Return':>11}")
    print(f"{'':{'<'}30}  {'':->8}  {'':->11}")
    summary: dict[str, dict] = {}
    for config_name, results in all_results.items():
        acc = sum(r["correct"] for r in results) / max(len(results), 1)
        ret = sum(r["episode_return"] for r in results) / max(len(results), 1)
        summary[config_name] = {"accuracy": round(acc, 3), "mean_return": round(ret, 3), "episodes": len(results)}
        print(f"  {config_name:<28}  {acc:>8.3f}  {ret:>+11.3f}")

    # Write output
    if output:
        out = {
            "summary": summary,
            "configs": {n: c for n, c in configs_to_test},
            "episodes": all_results,
        }
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nFull results → {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-training loader benchmark")
    parser.add_argument("--docs-dir", default="docs/", help="Path to org docs directory")
    parser.add_argument("--checkpoint", default="", help="Path to trained LoRA checkpoint (empty = heuristic fallback)")
    parser.add_argument("--env-url", default="http://localhost:8000")
    parser.add_argument("--tasks-file", "--task-bank", dest="tasks_file", default="all")
    parser.add_argument("--compare-slm", default="", help="SLM model ID for extraction comparison")
    parser.add_argument("--compare-llm", default="", help="Optional second model ID for extraction comparison")
    parser.add_argument("--output", default="rewards/loader_benchmark.json")
    args = parser.parse_args()

    asyncio.run(run_benchmark(
        docs_dir=Path(args.docs_dir),
        checkpoint=args.checkpoint,
        env_url=args.env_url,
        tasks_file=args.tasks_file,
        compare_slm=args.compare_slm,
        compare_llm=args.compare_llm,
        output=args.output,
    ))


if __name__ == "__main__":
    main()
