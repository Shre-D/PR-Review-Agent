from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .compat import CallToolAction, Observation, State

Verdict = Literal["approve", "request_changes", "reject", "escalate"]
AuthorLevel = Literal["junior", "mid", "senior", "lead", "non_tech", "unknown"]
FindingSeverity = Literal["critical", "warning", "info"]
RiskLevel = Literal["low", "medium", "high"]


class PRReviewAction(CallToolAction):
    """One environment action equals one tool invocation."""


class AuthorContext(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    level: AuthorLevel = "unknown"
    github_username: str = ""
    account_age_days: int = 0
    total_commits: int = 0
    recent_revert_rate: float = 0.0
    is_first_pr: bool = False
    team: str = ""


class Finding(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    tool_name: str = ""
    domain: str = ""
    severity: FindingSeverity = "info"
    message: str = ""
    path: str = ""
    line: int | None = None


class PRReviewVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    verdict: Verdict = "escalate"
    confidence: float = 0.0
    summary: str = ""
    critical_findings: list[Finding] = Field(default_factory=list)
    warnings: list[Finding] = Field(default_factory=list)
    suggestions: list[Finding] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    aggregate_score: float = 0.0
    review_context: str = ""


class CustomRule(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    domain: str = ""
    pattern: str = ""
    severity: FindingSeverity = "warning"
    message: str = ""
    languages: list[str] = Field(default_factory=list)


class ReviewConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    tool_weights: dict[str, float] = Field(default_factory=dict)
    enabled_tools: list[str] = Field(default_factory=list)
    planned_tools: list[str] = Field(default_factory=list)
    reject_threshold: float = 0.30
    request_changes_threshold: float = 0.65
    domain_priorities: dict[str, float] = Field(default_factory=dict)
    author_depth: dict[str, float] = Field(
        default_factory=lambda: {
            "junior": 1.4,
            "mid": 1.0,
            "senior": 0.7,
            "lead": 0.6,
            "non_tech": 1.2,
            "unknown": 1.0,
        }
    )
    critical_paths: list[str] = Field(default_factory=list)
    custom_rules: list[CustomRule] = Field(default_factory=list)
    architecture_summary: str = ""
    policy_model: str = "Qwen/Qwen3-1.7B"
    escalation_confidence_threshold: float = 0.6
    extraction_method: str = "structural"


class PRReviewObservation(Observation):
    diff_str: str = ""
    pr_description: str = ""
    primary_language: str = ""
    changed_file_types: list[str] = Field(default_factory=list)
    repo_kind: str = ""
    available_tools: list[str] = Field(default_factory=list)
    tools_called: list[str] = Field(default_factory=list)
    tool_results: dict[str, dict[str, Any]] = Field(default_factory=dict)
    review_history: list[str] = Field(default_factory=list)
    last_tool_name: str | None = None
    last_tool_result: dict[str, Any] = Field(default_factory=dict)
    task_id: str = ""
    step_count: int = 0
    author_context: AuthorContext = Field(default_factory=AuthorContext)
    critical_paths_touched: list[str] = Field(default_factory=list)
    estimated_risk_level: RiskLevel = "low"
    final_verdict: PRReviewVerdict | None = None


class PRReviewState(State):
    task_id: str = ""
    expected_verdict: str = ""
    primary_language: str = ""
    changed_file_types: list[str] = Field(default_factory=list)
    risk_domains: list[str] = Field(default_factory=list)
    tools_called_this_episode: list[str] = Field(default_factory=list)
    tool_results: dict[str, dict[str, Any]] = Field(default_factory=dict)
    review_history: list[str] = Field(default_factory=list)
    cumulative_reward: float = 0.0
