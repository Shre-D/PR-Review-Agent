from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolSpec:
    name: str
    display_name: str
    domains: tuple[str, ...]
    languages: tuple[str, ...]
    binary: str
    install_hint: str
    default_weight: float
    default_args: tuple[str, ...] = ()
    output_format: str = "text"
    status: str = "implemented"


@dataclass(frozen=True)
class ToolResult:
    tool_name: str
    score: float
    findings: tuple[str, ...] = ()
    severity_counts: dict[str, int] = field(default_factory=dict)
    raw: Any = None


TOOL_REGISTRY: dict[str, ToolSpec] = {
    "semgrep": ToolSpec(
        name="semgrep",
        display_name="Semgrep SAST",
        domains=("security",),
        languages=("all",),
        binary="semgrep",
        install_hint="pip install semgrep",
        default_weight=0.40,
        default_args=("--config=auto", "--json", "--quiet"),
        output_format="json",
    ),
    "ruff": ToolSpec(
        name="ruff",
        display_name="Ruff",
        domains=("quality",),
        languages=("python",),
        binary="ruff",
        install_hint="pip install ruff",
        default_weight=0.18,
        default_args=("check", "--output-format", "json"),
        output_format="json",
    ),
    "pylint": ToolSpec(
        name="pylint",
        display_name="Pylint",
        domains=("quality",),
        languages=("python",),
        binary="pylint",
        install_hint="pip install pylint",
        default_weight=0.16,
        default_args=("--output-format=json",),
        output_format="json",
    ),
    "radon": ToolSpec(
        name="radon",
        display_name="Radon",
        domains=("quality",),
        languages=("python",),
        binary="radon",
        install_hint="pip install radon",
        default_weight=0.10,
        default_args=("cc", "--json"),
        output_format="json",
    ),
    "pyyaml": ToolSpec(
        name="pyyaml",
        display_name="PyYAML parser",
        domains=("config",),
        languages=("yaml", "github_actions"),
        binary="python",
        install_hint="pip install pyyaml",
        default_weight=0.08,
        output_format="python",
    ),
    "go": ToolSpec(
        name="go",
        display_name="Go test",
        domains=("build", "tests"),
        languages=("go",),
        binary="go",
        install_hint="Install Go toolchain",
        default_weight=0.18,
        default_args=("test", "./..."),
    ),
    "javac": ToolSpec(
        name="javac",
        display_name="Java compiler",
        domains=("build",),
        languages=("java",),
        binary="javac",
        install_hint="Install JDK",
        default_weight=0.18,
    ),
    "cargo": ToolSpec(
        name="cargo",
        display_name="Cargo check",
        domains=("build",),
        languages=("rust",),
        binary="cargo",
        install_hint="Install Rust toolchain",
        default_weight=0.18,
        default_args=("check", "--quiet"),
    ),
    "tsc": ToolSpec(
        name="tsc",
        display_name="TypeScript compiler",
        domains=("build",),
        languages=("typescript", "javascript"),
        binary="tsc",
        install_hint="npm install -g typescript",
        default_weight=0.18,
        default_args=("--noEmit", "-p", "."),
    ),
    "bandit": ToolSpec(
        name="bandit",
        display_name="Bandit",
        domains=("security",),
        languages=("python",),
        binary="bandit",
        install_hint="pip install bandit",
        default_weight=0.30,
        output_format="json",
        status="planned",
    ),
    "gitleaks": ToolSpec(
        name="gitleaks",
        display_name="Gitleaks",
        domains=("security",),
        languages=("all",),
        binary="gitleaks",
        install_hint="Install gitleaks binary",
        default_weight=0.30,
        output_format="json",
        status="planned",
    ),
    "hadolint": ToolSpec(
        name="hadolint",
        display_name="Hadolint",
        domains=("config",),
        languages=("dockerfile",),
        binary="hadolint",
        install_hint="Install hadolint binary",
        default_weight=0.12,
        output_format="json",
        status="planned",
    ),
    "actionlint": ToolSpec(
        name="actionlint",
        display_name="actionlint",
        domains=("config", "build"),
        languages=("github_actions",),
        binary="actionlint",
        install_hint="Install actionlint binary",
        default_weight=0.12,
        status="planned",
    ),
    "yamllint": ToolSpec(
        name="yamllint",
        display_name="yamllint",
        domains=("config",),
        languages=("yaml", "github_actions"),
        binary="yamllint",
        install_hint="pip install yamllint",
        default_weight=0.08,
        status="planned",
    ),
    "mypy": ToolSpec(
        name="mypy",
        display_name="mypy",
        domains=("build", "quality"),
        languages=("python",),
        binary="mypy",
        install_hint="pip install mypy",
        default_weight=0.16,
        status="planned",
    ),
}


def implemented_tools() -> dict[str, ToolSpec]:
    return {name: spec for name, spec in TOOL_REGISTRY.items() if spec.status == "implemented"}


def tools_for_domain(domain: str, *, implemented_only: bool = True) -> list[ToolSpec]:
    registry = implemented_tools() if implemented_only else TOOL_REGISTRY
    return [spec for spec in registry.values() if domain in spec.domains]


def tools_for_language(language: str, *, implemented_only: bool = True) -> list[ToolSpec]:
    registry = implemented_tools() if implemented_only else TOOL_REGISTRY
    return [
        spec
        for spec in registry.values()
        if "all" in spec.languages or language in spec.languages
    ]


def build_domain_to_tools(registry: dict[str, ToolSpec] | None = None) -> dict[str, set[str]]:
    source = registry or implemented_tools()
    mapping: dict[str, set[str]] = {}
    for spec in source.values():
        for domain in spec.domains:
            mapping.setdefault(domain, set()).add(f"check_{domain}")
    return mapping


def normalize_findings(tool_name: str, payload: Any) -> ToolResult:
    if isinstance(payload, dict):
        score = payload.get("score", 1.0)
        findings = payload.get("findings", [])
        severity_counts = payload.get("severity_counts", {})
    else:
        score = 1.0
        findings = []
        severity_counts = {}

    try:
        normalized_score = max(0.0, min(1.0, float(score)))
    except (TypeError, ValueError):
        normalized_score = 0.0

    if not isinstance(findings, list):
        findings = [str(findings)]
    if not isinstance(severity_counts, dict):
        severity_counts = {}

    return ToolResult(
        tool_name=tool_name,
        score=normalized_score,
        findings=tuple(str(item) for item in findings),
        severity_counts={str(key): int(value) for key, value in severity_counts.items()},
        raw=payload,
    )
