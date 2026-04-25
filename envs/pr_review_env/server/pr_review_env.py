from __future__ import annotations

import random
import re
from typing import Any

from ..compat import CallToolObservation, MCPEnvironment, Observation, ensure_episode_id
from ..models import (
    AuthorContext,
    Finding,
    PRReviewAction,
    PRReviewObservation,
    PRReviewState,
    PRReviewVerdict,
    ReviewConfig,
)
from .context_loader import load_review_config
from .grader import outcome_summary, step_reward, terminal_reward
from .tasks import PRTask, get_random_task, get_task_by_id
from .tools import TOOL_NAMES, TOOL_REGISTRY, mcp


class PRReviewEnv(MCPEnvironment):
    """Multi-language PR review routing environment."""

    def __init__(
        self,
        task_path: str = "tasks/tasks.jsonl",
        seed: int | None = None,
        review_config: ReviewConfig | dict[str, Any] | None = None,
        review_config_path: str | None = None,
    ):
        super().__init__(mcp_server=mcp)
        self._tool_registry = TOOL_REGISTRY
        self._task_path = task_path
        self._rng = random.Random(seed)
        self._task: PRTask | None = None
        self._state = PRReviewState()
        self._done = False
        loaded_config = load_review_config(review_config_path) if review_config_path else review_config
        self._review_config = (
            loaded_config
            if isinstance(loaded_config, ReviewConfig) or loaded_config is None
            else ReviewConfig.model_validate(loaded_config)
        )

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        task_id: str | None = None,
        **kwargs: Any,
    ) -> PRReviewObservation:
        if seed is not None:
            self._rng.seed(seed)

        if task_id:
            self._task = get_task_by_id(task_id, self._task_path)
        else:
            self._task = get_random_task(self._task_path, self._rng)

        self._done = False
        self._state = PRReviewState(
            episode_id=ensure_episode_id(episode_id),
            task_id=self._task.task_id,
            expected_verdict=self._task.expected_verdict,
            primary_language=self._task.primary_language,
            changed_file_types=list(self._task.changed_file_types),
            risk_domains=list(self._task.risk_domains),
            tools_called_this_episode=[],
            tool_results={},
            review_history=[],
            cumulative_reward=0.0,
            step_count=0,
        )
        return self._observation(
            reward=0.0,
            last_tool_name=None,
            last_tool_result={},
        )

    @property
    def state(self) -> PRReviewState:
        return self._state

    def step(
        self,
        action: PRReviewAction,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> PRReviewObservation:
        if self._task is None:
            raise RuntimeError("reset() must be called before step()")
        if self._done:
            raise RuntimeError("episode already finished; call reset()")

        if action.tool_name not in {"submit_review", "escalate"}:
            action.arguments.setdefault("task_id", self._task.task_id)
            if self._review_config is not None:
                action.arguments.setdefault("review_config", self._review_config.model_dump())

        already_called = action.tool_name in self._state.tools_called_this_episode
        base_obs = super().step(action, timeout_s=timeout_s, **kwargs)
        if not isinstance(base_obs, CallToolObservation):
            raise TypeError(f"Unexpected observation type: {type(base_obs)}")

        self._state.step_count += 1
        self._state.tools_called_this_episode.append(action.tool_name)

        result_payload = self._extract_result_payload(base_obs)
        history_line = action.tool_name
        if base_obs.error is not None:
            result_payload = {
                "tool": action.tool_name,
                "score": 0.0,
                "error": base_obs.error.message,
            }
            history_line = f"{action.tool_name}: error={base_obs.error.message}"
        else:
            self._state.tool_results[action.tool_name] = result_payload
            score = result_payload.get("score")
            history_line = f"{action.tool_name}: score={score}"

        self._state.review_history.append(history_line)
        reward = step_reward(
            self._task,
            action.tool_name,
            already_called=already_called,
            step_count=self._state.step_count,
            config=self._review_config,
        )

        final_verdict = None
        if action.tool_name in {"submit_review", "escalate"}:
            submitted_verdict = result_payload.get("verdict", action.tool_name)
            confidence = result_payload.get("confidence")
            reward += terminal_reward(
                self._task,
                submitted_verdict,
                self._state.tool_results,
                config=self._review_config,
                confidence=confidence,
            )
            final_verdict = self._build_final_verdict(result_payload)
            self._done = True
            self._state.review_history.append(
                f"terminal verdict={submitted_verdict} expected={self._task.expected_verdict}"
            )

        self._state.cumulative_reward = round(self._state.cumulative_reward + reward, 3)
        return self._observation(
            reward=round(reward, 3),
            last_tool_name=action.tool_name,
            last_tool_result=result_payload,
            final_verdict=final_verdict,
        )

    def _extract_result_payload(self, observation: CallToolObservation) -> dict[str, Any]:
        result = observation.result
        if result is None:
            return {}
        if isinstance(result, dict):
            return result
        if hasattr(result, "data"):
            data = getattr(result, "data")
            if isinstance(data, dict):
                return data
        if hasattr(result, "structured_content"):
            structured = getattr(result, "structured_content")
            if isinstance(structured, dict) and isinstance(structured.get("result"), dict):
                return structured["result"]
        return {"raw_result": str(result)}

    def _observation(
        self,
        reward: float,
        last_tool_name: str | None,
        last_tool_result: dict[str, Any],
        final_verdict: PRReviewVerdict | None = None,
    ) -> PRReviewObservation:
        assert self._task is not None
        critical_paths = self._critical_paths_touched(self._task.diff_str)
        return PRReviewObservation(
            diff_str=self._task.diff_str,
            pr_description=self._task.pr_description,
            primary_language=self._task.primary_language,
            changed_file_types=list(self._task.changed_file_types),
            repo_kind=self._task.repo_kind,
            available_tools=list(TOOL_NAMES),
            tools_called=list(self._state.tools_called_this_episode),
            tool_results=dict(self._state.tool_results),
            review_history=list(self._state.review_history),
            last_tool_name=last_tool_name,
            last_tool_result=last_tool_result,
            task_id=self._task.task_id,
            step_count=self._state.step_count,
            author_context=AuthorContext(level=getattr(self._task, "author_level", "mid")),
            critical_paths_touched=critical_paths,
            estimated_risk_level=self._estimated_risk_level(critical_paths),
            final_verdict=final_verdict,
            done=self._done,
            reward=reward,
            metadata=outcome_summary(
                self._task,
                self._state.tool_results,
                submitted_verdict=last_tool_result.get("verdict"),
                config=self._review_config,
            ),
        )

    def _critical_paths_touched(self, diff_str: str) -> list[str]:
        if not self._review_config:
            return []
        paths = [
            new if new != "/dev/null" else old
            for old, new in re.findall(r"^diff --git a/(.*?) b/(.*?)$", diff_str, flags=re.MULTILINE)
        ]
        touched: list[str] = []
        for critical_path in self._review_config.critical_paths:
            normalized = critical_path.rstrip("/")
            if any(path == normalized or path.startswith(f"{normalized}/") for path in paths):
                touched.append(critical_path)
        return touched

    def _estimated_risk_level(self, critical_paths: list[str]) -> str:
        assert self._task is not None
        if critical_paths or {"security", "config"} & set(self._task.risk_domains):
            return "high"
        if self._task.risk_domains or len(self._task.changed_file_types) > 1:
            return "medium"
        return "low"

    def _build_final_verdict(self, result_payload: dict[str, Any]) -> PRReviewVerdict:
        assert self._task is not None
        findings: list[Finding] = []
        for tool_name, payload in self._state.tool_results.items():
            if tool_name in {"submit_review", "escalate"}:
                continue
            for message in payload.get("findings", [])[:10]:
                score = payload.get("score", 1.0)
                severity = "critical" if score <= 0.55 else "warning"
                domain = tool_name.replace("check_", "")
                findings.append(
                    Finding(
                        tool_name=tool_name,
                        domain=domain,
                        severity=severity,
                        message=str(message),
                    )
                )

        critical = [finding for finding in findings if finding.severity == "critical"]
        warnings = [finding for finding in findings if finding.severity == "warning"]
        aggregate_score = outcome_summary(
            self._task,
            self._state.tool_results,
            submitted_verdict=result_payload.get("verdict"),
            config=self._review_config,
        )["aggregate_score"]

        confidence = result_payload.get("confidence")
        try:
            normalized_confidence = max(0.0, min(1.0, float(confidence)))
        except (TypeError, ValueError):
            normalized_confidence = 0.0

        return PRReviewVerdict(
            verdict=result_payload.get("verdict", "escalate"),
            confidence=normalized_confidence,
            summary=result_payload.get("reasoning", ""),
            critical_findings=critical,
            warnings=warnings,
            tools_used=[
                tool for tool in self._state.tools_called_this_episode if tool not in {"submit_review", "escalate"}
            ],
            aggregate_score=aggregate_score,
            review_context=f"risk={self._estimated_risk_level(self._critical_paths_touched(self._task.diff_str))}",
        )

    def _step_impl(
        self,
        action: Any,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> Observation:
        raise TypeError(
            "PRReviewEnv only accepts tool-call actions. Use PRReviewAction with a tool name."
        )
