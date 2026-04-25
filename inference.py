from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.pr_review_env.server.pr_review_env import PRReviewEnv
from envs.pr_review_env.server.tasks import load_tasks, task_review_config
from benchmarks.run_baselines import heuristic_policy, decide_final_verdict
from train.grpo_train import (
    SYSTEM_PROMPT, 
    build_obs_prompt, 
    parse_action, 
    _action_with_state_args
)
from train.adaptive_router import review_requirements

OLLAMA_PREFIX = "ollama:"
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
BENCHMARK_FIELDS = {"expected_verdict", "correct"}


def is_ollama_model(model_arg: str) -> bool:
    return model_arg.lower().startswith(OLLAMA_PREFIX)


def ollama_model_name(model_arg: str) -> str:
    return model_arg[len(OLLAMA_PREFIX):] if is_ollama_model(model_arg) else model_arg


def load_model(model_id: str, checkpoint: str | None = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"Loading model: {model_id}...", file=sys.stderr)
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Use 4-bit quantization if possible for local runs
    try:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            bnb_4bit_use_double_quant=True,
        )
    except ImportError:
        bnb_config = None

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True
    )

    if checkpoint:
        print(f"Loading LoRA checkpoint: {checkpoint}...", file=sys.stderr)
        model = PeftModel.from_pretrained(model, checkpoint)

    model.eval()
    return model, tokenizer


def build_messages(obs):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_obs_prompt(obs)}
    ]


def run_qwen_step(model, tokenizer, obs):
    import torch
    
    messages = build_messages(obs)
    
    input_ids = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        return_tensors="pt"
    ).to(next(model.parameters()).device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][input_ids.shape[-1]:]
    response_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return parse_action(response_text)


def run_ollama_step(model_name: str, obs, host: str | None = None):
    host = (host or os.getenv("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST).rstrip("/")
    payload = {
        "model": model_name,
        "messages": build_messages(obs),
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": 128,
        },
    }
    request = urllib.request.Request(
        f"{host}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not reach Ollama at {host}. Start it with `ollama serve` "
            f"and make sure `{model_name}` is pulled."
        ) from exc

    response_text = result.get("message", {}).get("content") or result.get("response") or ""
    return parse_action(response_text.strip())


def _action_for_log(action, hide_diff: bool) -> dict:
    payload = action.model_dump()
    if hide_diff and isinstance(payload.get("arguments"), dict) and "diff_str" in payload["arguments"]:
        payload["arguments"] = dict(payload["arguments"])
        payload["arguments"]["diff_str"] = "<hidden>"
    return payload


def run_inference():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.getenv("MODEL_NAME", "heuristic"))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--limit", type=int, default=int(os.getenv("TASK_LIMIT", "3")))
    parser.add_argument("--output-format", default="trace", choices=["trace", "packet", "jsonl"])
    parser.add_argument("--show-expected", action="store_true")
    parser.add_argument("--hide-diff", action="store_true")
    args = parser.parse_args()

    env_name = "pr-review-env"
    
    # Initialize environment
    tasks = load_tasks()
    tasks_to_run = tasks[:args.limit] if args.limit > 0 else tasks

    is_heuristic = args.model.lower() == "heuristic"
    is_ollama = is_ollama_model(args.model)
    model, tokenizer = None, None

    if is_ollama and args.checkpoint:
        raise ValueError("Ollama inference cannot load a PEFT/LoRA checkpoint; omit --checkpoint.")

    if not is_heuristic and not is_ollama:
        model, tokenizer = load_model(args.model, args.checkpoint)
    elif is_ollama:
        print(
            f"Using Ollama model: {ollama_model_name(args.model)} "
            f"at {(os.getenv('OLLAMA_HOST') or DEFAULT_OLLAMA_HOST).rstrip('/')}",
            file=sys.stderr,
        )

    for task in tasks_to_run:
        task_config = task_review_config(task, mode="short")
        env = PRReviewEnv(review_config=task_config)
        # 1. Emit [START] metadata
        observation = env.reset(task_id=task.task_id)
        route = review_requirements(observation, task_config)
        if args.output_format == "trace":
            print(
                f"[START] task={task.task_id} env={env_name} model={args.model} "
                f"tier={route['tier']} max_steps={route['max_steps']} min_tools={route['min_tools']}"
            )
            sys.stdout.flush()
        step_count = 0
        max_steps = int(route["max_steps"])
        forced_submit = False
        heuristic_actions = heuristic_policy(observation) if is_heuristic else []
        heuristic_index = 0

        while not observation.done and step_count < max_steps:
            step_count += 1
            if is_heuristic:
                if heuristic_index < len(heuristic_actions):
                    action = _action_with_state_args(heuristic_actions[heuristic_index], observation)
                    heuristic_index += 1
                else:
                    action = decide_final_verdict(observation)
            else:
                if is_ollama:
                    raw_action = run_ollama_step(ollama_model_name(args.model), observation)
                else:
                    raw_action = run_qwen_step(model, tokenizer, observation)
                action = _action_with_state_args(raw_action, observation)

            observation = env.step(action)
            reward = observation.reward or 0.0
            if args.output_format == "trace":
                print(f"[STEP] step={step_count} action={json.dumps(_action_for_log(action, args.hide_diff))} reward={reward}")
                sys.stdout.flush()

        # Ensure we eventually submit if model didn't
        if not observation.done:
            forced_submit = True
            step_count += 1
            final_action = decide_final_verdict(observation)
            observation = env.step(final_action)
            if args.output_format == "trace":
                reward = observation.reward or 0.0
                print(
                    f"[STEP] step={step_count} action={json.dumps(_action_for_log(final_action, args.hide_diff))} "
                    f"reward={reward} forced_submit=1"
                )
                sys.stdout.flush()

        # Final verdict check
        # The grader handles the score in observation.reward for terminal actions
        # but [END] score is typically the task success (0.0 to 1.0)
        # Based on benchmark logic:
        is_correct = 1 if observation.final_verdict and observation.final_verdict.verdict == task.expected_verdict else 0
        terminal_reward = observation.reward or 0.0
        episode_return = env.state.cumulative_reward
        if args.output_format == "trace":
            extra = ""
            if args.show_expected:
                extra = f" expected={task.expected_verdict} predicted={observation.final_verdict.verdict if observation.final_verdict else 'unknown'}"
            print(
                f"[END] correct={is_correct} episode_return={episode_return} terminal_reward={terminal_reward} "
                f"steps={step_count} forced_submit={int(forced_submit)}{extra}"
            )
            sys.stdout.flush()
        else:
            packet = build_review_packet(
                task,
                observation,
                episode_return=episode_return,
                model_name=args.model,
                steps_used=step_count,
                route=route,
                include_expected=args.show_expected,
            )
            if args.output_format == "jsonl":
                print(json.dumps(packet))
            else:
                print(render_packet(packet))


def build_review_packet(task, observation, episode_return: float, model_name: str, steps_used: int, route: dict, include_expected: bool = False) -> dict:
    findings = []
    for tool_name, payload in observation.tool_results.items():
        for message in payload.get("findings", [])[:3]:
            findings.append(
                {
                    "tool": tool_name,
                    "severity": "critical" if float(payload.get("score", 1.0)) <= 0.55 else "warning",
                    "message": str(message),
                }
            )
    non_terminal_tools = [tool for tool in observation.tools_called if tool not in {"submit_review", "escalate"}]
    duplicate_calls = max(0, len(non_terminal_tools) - len(set(non_terminal_tools)))
    packet = {
        "task_id": task.task_id,
        "model": model_name,
        "predicted_verdict": observation.final_verdict.verdict if observation.final_verdict else "unknown",
        "risk": observation.estimated_risk_level,
        "route": {
            "tier": route["tier"],
            "min_tools": route["min_tools"],
            "max_steps": route["max_steps"],
            "steps_used": steps_used,
            "remaining_steps": max(0, route["max_steps"] - steps_used),
        },
        "reward": {
            "episode_return": round(float(episode_return), 4),
            "terminal_reward": round(float(observation.reward or 0.0), 4),
        },
        "tool_cost": {
            "total_calls": len(non_terminal_tools),
            "unique_tools": len(set(non_terminal_tools)),
            "duplicate_calls": duplicate_calls,
        },
        "tools_used": non_terminal_tools,
        "findings": findings,
        "evidence_summary": observation.final_verdict.summary if observation.final_verdict else "",
        "escalation_reason": observation.last_tool_result.get("reason", "") if observation.last_tool_name == "escalate" else "",
    }
    if include_expected:
        packet["expected_verdict"] = task.expected_verdict
        packet["correct"] = bool(observation.final_verdict and observation.final_verdict.verdict == task.expected_verdict)
    return packet


def render_packet(packet: dict) -> str:
    lines = [
        f"PR: {packet['task_id']}",
        f"Model: {packet['model']}",
        f"Risk: {packet['risk']}",
        f"Route: {packet['route']['tier']}, {packet['route']['steps_used']}/{packet['route']['max_steps']} steps",
        f"Decision: {packet['predicted_verdict']}",
        "Evidence:",
    ]
    for finding in packet["findings"][:5]:
        lines.append(f"- {finding['tool']}: {finding['message']}")
    if not packet["findings"]:
        lines.append("- no findings")
    lines.append(
        f"Tool cost: {packet['tool_cost']['total_calls']} calls, {packet['tool_cost']['duplicate_calls']} duplicates"
    )
    lines.append(f"Episode return: {packet['reward']['episode_return']}")
    return "\n".join(lines)

if __name__ == "__main__":
    try:
        run_inference()
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
