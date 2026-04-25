from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import yaml

from ..models import ReviewConfig

try:
    from fastmcp import FastMCP
except Exception:  # pragma: no cover - used only when fastmcp is unavailable
    class FastMCP:  # type: ignore[override]
        def __init__(self, name: str):
            self.name = name
            self._tools: dict[str, Any] = {}

        def tool(self, fn=None, **kwargs):
            def decorator(func):
                self._tools[func.__name__] = func
                return func

            if fn is not None:
                return decorator(fn)
            return decorator

        def get_tools(self):
            return self._tools


from .fixtures import changed_files_in_workspace, has_fixture, materialize_workspace


REPO_ROOT = Path(__file__).resolve().parents[3]
mcp = FastMCP("pr-review-tools")

TOOL_NAMES = [
    "check_security",
    "check_quality",
    "check_build_and_types",
    "check_tests",
    "check_config",
    "submit_review",
    "escalate",
]

SUPPORTED_LANGUAGES = {"python", "typescript", "javascript", "java", "go", "rust"}
REPO_GLOBAL_FILE_TYPES = {"dockerfile", "yaml", "gitignore", "github_actions"}


def _backend_mode() -> str:
    return os.getenv("PR_REVIEW_TOOL_BACKEND", "hybrid").strip().lower()


def _resolve_executable(name: str) -> str | None:
    for candidate in [
        REPO_ROOT / ".venv" / "bin" / name,
        REPO_ROOT / ".tools" / "bin" / name,
    ]:
        if candidate.exists():
            return str(candidate)
    return shutil.which(name)


def _changed_paths(diff_str: str) -> list[tuple[str, str]]:
    return re.findall(r"^diff --git a/(.*?) b/(.*?)$", diff_str, flags=re.MULTILINE)


def _flatten_paths(diff_str: str) -> list[str]:
    paths = []
    for old_path, new_path in _changed_paths(diff_str):
        chosen = new_path if new_path != "/dev/null" else old_path
        paths.append(chosen)
    return paths


def _detect_file_types(paths: list[str]) -> list[str]:
    detected: set[str] = set()
    for path in paths:
        lower = path.lower()
        suffix = Path(lower).suffix
        if lower.endswith(".py"):
            detected.add("python")
        elif lower.endswith((".ts", ".tsx")):
            detected.add("typescript")
        elif lower.endswith((".js", ".jsx")):
            detected.add("javascript")
        elif lower.endswith(".java"):
            detected.add("java")
        elif lower.endswith(".go"):
            detected.add("go")
        elif lower.endswith(".rs"):
            detected.add("rust")
        elif lower.endswith(("dockerfile", "/dockerfile")) or Path(lower).name == "dockerfile":
            detected.add("dockerfile")
        elif lower.endswith((".yml", ".yaml")):
            detected.add("yaml")
            if ".github/workflows/" in lower:
                detected.add("github_actions")
        elif Path(lower).name == ".gitignore":
            detected.add("gitignore")
        elif suffix in {".toml", ".xml", ".gradle"} or Path(lower).name in {
            "package.json",
            "pom.xml",
            "cargo.toml",
            "go.mod",
        }:
            detected.add("build_manifest")
    return sorted(detected)


def _added_lines(diff_str: str) -> list[str]:
    lines: list[str] = []
    for line in diff_str.splitlines():
        if line.startswith("+++") or line.startswith("diff --git") or line.startswith("@@"):
            continue
        if line.startswith("+"):
            lines.append(line[1:])
    return lines


def _candidate_source(diff_str: str) -> str:
    return "\n".join(_added_lines(diff_str)).strip()


def _language_suffix(diff_str: str) -> str:
    file_types = _detect_file_types(_flatten_paths(diff_str))
    if "python" in file_types:
        return ".py"
    if "typescript" in file_types:
        return ".ts"
    if "javascript" in file_types:
        return ".js"
    if "java" in file_types:
        return ".java"
    if "go" in file_types:
        return ".go"
    if "rust" in file_types:
        return ".rs"
    if "yaml" in file_types:
        return ".yaml"
    if "dockerfile" in file_types:
        return ".Dockerfile"
    return ".txt"


def _temp_source_path(diff_str: str, suffix: str) -> Path | None:
    source = _candidate_source(diff_str)
    if not source:
        return None

    handle = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=suffix,
        delete=False,
        encoding="utf-8",
    )
    with handle:
        handle.write(source)
        handle.write("\n")
    return Path(handle.name)


@contextmanager
def _analysis_targets(diff_str: str, task_id: str = "") -> Iterator[dict[str, Any]]:
    if task_id and has_fixture(task_id):
        with materialize_workspace(task_id) as workspace:
            targets = changed_files_in_workspace(workspace, diff_str)
            yield {
                "analysis_mode": "fixture_backed",
                "workspace": workspace,
                "targets": targets,
            }
        return

    temp_path = _temp_source_path(diff_str, _language_suffix(diff_str))
    if temp_path is None:
        yield {
            "analysis_mode": "heuristic_only",
            "workspace": None,
            "targets": [],
        }
        return

    try:
        yield {
            "analysis_mode": "diff_reconstructed",
            "workspace": temp_path.parent,
            "targets": [temp_path],
        }
    finally:
        temp_path.unlink(missing_ok=True)


def _run_command(
    command: list[str],
    cwd: Path | None = None,
    timeout_s: int = 20,
) -> dict[str, Any]:
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
        cwd=str(cwd) if cwd else None,
    )
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def _run_json_command(
    command: list[str],
    cwd: Path | None = None,
    timeout_s: int = 20,
) -> dict[str, Any] | list[Any]:
    result = _run_command(command, cwd=cwd, timeout_s=timeout_s)
    stdout = result["stdout"]
    if not stdout:
        return {}
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        return result


def _heuristic_findings(diff_str: str) -> dict[str, list[str]]:
    lower = diff_str.lower()
    findings: dict[str, list[str]] = {
        "security": [],
        "quality": [],
        "build": [],
        "tests": [],
        "config": [],
    }

    security_patterns = {
        "Potential SQL injection via string-built query": [
            "select *",
            "request.args",
            "execute(sql",
        ],
        "Dynamic code execution via eval": ["eval("],
        "Unsafe Java runtime exec detected": ["runtime.getruntime().exec("],
        "Shell command injection risk (.arg -c pattern)": [".arg(\"-c\")"],
        "Unsafe pickle deserialization": ["pickle.loads("],
        "Possible hardcoded credential": ["api_token", "password =", "token ="],
    }
    for finding, patterns in security_patterns.items():
        if all(pattern in lower for pattern in patterns):
            findings["security"].append(finding)

    if "unwrap()" in lower or ".getprofile().getdisplayname()" in lower:
        findings["quality"].append("Unsafe error/null handling likely to fail in production.")
    if lower.count(" if ") + lower.count(" if(") >= 4:
        findings["quality"].append("Diff appears overly branch-heavy and may need simplification.")
    if "todo" in lower or "fixme" in lower:
        findings["quality"].append("Introduces TODO/FIXME markers in production code.")

    if any(token in lower for token in ["package.json", "pom.xml", "cargo.toml", "go.mod"]):
        findings["build"].append("Build manifest changed; review dependency and compatibility impact.")
    if ".github/workflows/" in lower and "permissions:" in lower and "write" in lower:
        findings["build"].append("Workflow permissions look broader than least privilege.")

    if any(token in lower for token in ["def ", "function ", "public ", "func ", "pub fn "]) and "test" not in lower:
        findings["tests"].append("Behavior changed without visible test coverage in the diff.")

    if "from python:" in lower or "from node:" in lower:
        findings["config"].append("Base image is unpinned; consider digest or narrower version pinning.")
    if "cmd [" in lower and "user " not in lower:
        findings["config"].append("Container runs without an explicit non-root user.")
    if "log_level" in lower and "debug" in lower:
        findings["config"].append("Production config enables debug logging.")
    if ".github/workflows/" in lower and "permissions:" in lower and "write" in lower:
        findings["config"].append("Workflow permissions look broader than least privilege.")
    removed_lines = [
        line[1:].strip()
        for line in diff_str.splitlines()
        if line.startswith("-") and not line.startswith("---")
    ]
    if ".gitignore" in lower and any(
        ".env" in rl or "secret" in rl or "*.pem" in rl or "*.key" in rl
        for rl in removed_lines
    ):
        findings["config"].append(
            "Gitignore narrowing removes broad env/secret protection — files may become trackable."
        )
    elif ".env.production" in lower or "secrets/" in lower:
        findings["config"].append("Ignore rules may hide deployment secrets or critical config.")

    return findings


def _score_from_findings(findings: list[str], base: float = 1.0, penalty: float = 0.18) -> float:
    return max(0.0, round(base - penalty * len(findings), 3))


def _coerce_review_config(review_config: dict[str, Any] | ReviewConfig | None) -> ReviewConfig | None:
    if review_config is None:
        return None
    if isinstance(review_config, ReviewConfig):
        return review_config
    return ReviewConfig.model_validate(review_config)


def _apply_custom_rules(
    diff_str: str,
    domain: str,
    review_config: dict[str, Any] | ReviewConfig | None = None,
) -> list[str]:
    config = _coerce_review_config(review_config)
    if config is None:
        return []

    findings: list[str] = []
    for rule in config.custom_rules:
        if rule.domain != domain:
            continue
        try:
            matched = re.search(rule.pattern, diff_str, flags=re.IGNORECASE | re.MULTILINE)
        except re.error:
            matched = rule.pattern.lower() in diff_str.lower()
        if matched:
            findings.append(f"[{rule.severity.upper()}] {rule.message}")
    return findings


def _base_result(tool: str, diff_str: str, task_id: str = "") -> dict[str, Any]:
    paths = _flatten_paths(diff_str)
    file_types = _detect_file_types(paths)
    primary_language = next(
        (item for item in file_types if item in SUPPORTED_LANGUAGES),
        file_types[0] if file_types else "unknown",
    )
    return {
        "tool": tool,
        "task_id": task_id,
        "changed_paths": paths,
        "file_types": file_types,
        "primary_language": primary_language,
        "backend": _backend_mode(),
    }


def _semgrep_scan(targets: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    semgrep = _resolve_executable("semgrep")
    if _backend_mode() == "heuristic" or semgrep is None or not targets["targets"]:
        return {}, []

    payload = _run_json_command(
        [semgrep, "--config=auto", "--json", "--quiet", *[str(path) for path in targets["targets"]]],
        cwd=targets["workspace"],
        timeout_s=20,
    )
    if not isinstance(payload, dict):
        return {}, []

    findings = []
    for finding in payload.get("results", [])[:10]:
        message = finding.get("extra", {}).get("message")
        if message:
            findings.append(message)
    return payload, findings


def _python_quality_scan(targets: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    findings: list[str] = []
    payload: dict[str, Any] = {}
    target_files = [path for path in targets["targets"] if path.suffix == ".py"]
    if not target_files:
        return payload, findings

    ruff = _resolve_executable("ruff")
    if _backend_mode() != "heuristic" and ruff is not None:
        ruff_payload = _run_json_command(
            [ruff, "check", "--output-format", "json", *[str(path) for path in target_files]],
            cwd=targets["workspace"],
            timeout_s=20,
        )
        payload["ruff"] = ruff_payload
        if isinstance(ruff_payload, list) and ruff_payload:
            findings.append("Ruff reported style or correctness issues in the candidate patch.")

    pylint = _resolve_executable("pylint")
    if _backend_mode() != "heuristic" and pylint is not None:
        pylint_payload = _run_json_command(
            [pylint, "--output-format=json", *[str(path) for path in target_files]],
            cwd=targets["workspace"],
            timeout_s=20,
        )
        payload["pylint"] = pylint_payload
        if isinstance(pylint_payload, list) and pylint_payload:
            findings.append("Pylint reported issues in the candidate patch.")

    radon = _resolve_executable("radon")
    if _backend_mode() != "heuristic" and radon is not None:
        radon_payload = _run_json_command(
            [radon, "cc", "--json", *[str(path) for path in target_files]],
            cwd=targets["workspace"],
            timeout_s=20,
        )
        payload["radon"] = radon_payload
        if isinstance(radon_payload, dict) and radon_payload:
            findings.append("Radon reported non-trivial cyclomatic complexity in the patch.")

    return payload, findings


def _build_and_type_findings(targets: dict[str, Any], file_types: list[str]) -> tuple[dict[str, Any], list[str], list[str]]:
    details: dict[str, Any] = {}
    findings: list[str] = []
    tools_used: list[str] = []

    if _backend_mode() == "heuristic" or not targets["targets"]:
        return details, findings, tools_used

    workspace = targets["workspace"]

    if "typescript" in file_types or "javascript" in file_types:
        tsc = _resolve_executable("tsc")
        pyright = _resolve_executable("pyright")
        ts_targets = [path for path in targets["targets"] if path.suffix in {".ts", ".tsx", ".js", ".jsx"}]
        if tsc is not None and workspace is not None:
            details["tsc"] = _run_command([tsc, "--noEmit", "-p", "."], cwd=workspace, timeout_s=20)
            tools_used.append("tsc")
            if details["tsc"]["returncode"] != 0:
                findings.append("TypeScript compilation failed for the fixture workspace.")
        elif pyright is not None and ts_targets:
            details["pyright"] = _run_command(
                [pyright, "--outputjson", *[str(p) for p in ts_targets]],
                cwd=workspace,
                timeout_s=20,
            )
            tools_used.append("pyright")
            if details["pyright"]["returncode"] != 0:
                findings.append("Pyright reported type errors in the TypeScript/JavaScript files.")

    if "java" in file_types:
        javac = _resolve_executable("javac")
        java_targets = [path for path in targets["targets"] if path.suffix == ".java"]
        if javac is not None and java_targets:
            out_dir = workspace / ".javac-out"
            out_dir.mkdir(exist_ok=True)
            details["javac"] = _run_command(
                [javac, "-d", str(out_dir), *[str(path) for path in java_targets]],
                cwd=workspace,
                timeout_s=20,
            )
            tools_used.append("javac")
            if details["javac"]["returncode"] != 0:
                findings.append("Java compilation failed for the fixture workspace.")

    if "go" in file_types:
        go = _resolve_executable("go")
        if go is not None and workspace is not None:
            details["go_test"] = _run_command([go, "test", "./..."], cwd=workspace, timeout_s=20)
            tools_used.append("go")
            if details["go_test"]["returncode"] != 0:
                findings.append("Go tests or compilation failed for the fixture workspace.")

    if "rust" in file_types:
        cargo = _resolve_executable("cargo")
        if cargo is not None and workspace is not None:
            details["cargo_check"] = _run_command([cargo, "check", "--quiet"], cwd=workspace, timeout_s=20)
            tools_used.append("cargo")
            if details["cargo_check"]["returncode"] != 0:
                findings.append("Cargo check failed for the fixture workspace.")

    return details, findings, tools_used


def _yaml_findings(targets: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    details: dict[str, Any] = {"parsed_files": [], "errors": []}
    findings: list[str] = []
    for path in targets["targets"]:
        if path.suffix not in {".yml", ".yaml"}:
            continue
        try:
            yaml.safe_load(path.read_text(encoding="utf-8"))
            details["parsed_files"].append(str(path))
        except yaml.YAMLError as exc:
            details["errors"].append({"path": str(path), "error": str(exc)})
            findings.append(f"YAML parse failed for {path.name}.")
    return details, findings


def _dockerfile_findings(targets: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    details: dict[str, Any] = {"inspected_files": []}
    findings: list[str] = []
    for path in targets["targets"]:
        if path.name.lower() != "dockerfile":
            continue
        details["inspected_files"].append(str(path))
        text = path.read_text(encoding="utf-8").lower()
        if "from " in text and ":latest" in text:
            findings.append("Dockerfile uses an overly broad base image tag.")
        if "user " not in text:
            findings.append("Dockerfile does not switch to a non-root user.")
    return details, findings


@mcp.tool
def check_security(
    diff_str: str,
    task_id: str = "",
    review_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run security-focused review across supported code and config diffs."""
    result = _base_result("check_security", diff_str, task_id=task_id)
    heuristics = _heuristic_findings(diff_str)["security"]

    with _analysis_targets(diff_str, task_id=task_id) as targets:
        semgrep_payload, semgrep_findings = _semgrep_scan(targets)
        combined = list(heuristics)
        combined.extend(semgrep_findings)
        combined.extend(_apply_custom_rules(diff_str, "security", review_config))
        result.update(
            {
                "analysis_mode": targets["analysis_mode"],
                "score": _score_from_findings(combined, penalty=0.45),
                "findings": combined,
                "semgrep_findings": len(semgrep_payload.get("results", [])) if semgrep_payload else 0,
                "real_tools_used": ["semgrep"] if semgrep_payload else [],
            }
        )
    return result


@mcp.tool
def check_quality(
    diff_str: str,
    task_id: str = "",
    review_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Review maintainability, error handling, and complexity drift."""
    result = _base_result("check_quality", diff_str, task_id=task_id)
    heuristics = _heuristic_findings(diff_str)["quality"]

    with _analysis_targets(diff_str, task_id=task_id) as targets:
        payload, tool_findings = _python_quality_scan(targets)
        combined = list(heuristics)
        combined.extend(tool_findings)
        combined.extend(_apply_custom_rules(diff_str, "quality", review_config))
        real_tools_used: list[str] = []
        for name in ("ruff", "pylint", "radon"):
            if payload.get(name):
                real_tools_used.append(name)

        result.update(
            {
                "analysis_mode": targets["analysis_mode"],
                "score": _score_from_findings(combined, penalty=0.16),
                "findings": combined,
                "quality_tooling": payload,
                "real_tools_used": real_tools_used,
            }
        )
    return result


@mcp.tool
def check_build_and_types(
    diff_str: str,
    task_id: str = "",
    review_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Inspect build, type, dependency, and workflow compatibility risk."""
    result = _base_result("check_build_and_types", diff_str, task_id=task_id)
    heuristics = _heuristic_findings(diff_str)["build"]

    with _analysis_targets(diff_str, task_id=task_id) as targets:
        build_details, build_findings, tools_used = _build_and_type_findings(
            targets,
            result["file_types"],
        )
        combined = list(heuristics)
        combined.extend(build_findings)
        combined.extend(_apply_custom_rules(diff_str, "build", review_config))
        if any(item in result["file_types"] for item in ["typescript", "java", "go", "rust"]):
            combined.append("Language change may require build or type validation before merge.")
        if "build_manifest" in result["file_types"]:
            combined.append("Dependency or build manifest changed.")

        result.update(
            {
                "analysis_mode": targets["analysis_mode"],
                "score": _score_from_findings(combined, penalty=0.14),
                "findings": combined,
                "build_tooling": build_details,
                "real_tools_used": tools_used,
            }
        )
    return result


@mcp.tool
def check_tests(
    diff_str: str,
    task_id: str = "",
    review_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Estimate whether the PR includes enough verification for changed behavior."""
    result = _base_result("check_tests", diff_str, task_id=task_id)
    findings = _heuristic_findings(diff_str)["tests"]
    findings.extend(_apply_custom_rules(diff_str, "tests", review_config))

    with _analysis_targets(diff_str, task_id=task_id) as targets:
        workspace = targets["workspace"]
        if workspace is not None and targets["analysis_mode"] == "fixture_backed":
            test_files = [
                path
                for path in workspace.rglob("*")
                if path.is_file() and ("test" in path.name.lower() or "spec" in path.name.lower())
            ]
        else:
            test_files = []

        if test_files:
            findings = [
                finding
                for finding in findings
                if "without visible test coverage" not in finding
            ]
        elif any(item in result["file_types"] for item in REPO_GLOBAL_FILE_TYPES):
            findings.append("Config-only changes still need CI confidence or rollback clarity.")

        result.update(
            {
                "analysis_mode": targets["analysis_mode"],
                "score": _score_from_findings(findings, penalty=0.12),
                "findings": findings,
                "real_tools_used": ["workspace_test_discovery"] if workspace is not None else [],
                "test_files_detected": len(test_files),
            }
        )
    return result


@mcp.tool
def check_config(
    diff_str: str,
    task_id: str = "",
    review_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Inspect Dockerfile, YAML, workflow, and ignore-rule changes."""
    result = _base_result("check_config", diff_str, task_id=task_id)
    heuristics = _heuristic_findings(diff_str)["config"]

    with _analysis_targets(diff_str, task_id=task_id) as targets:
        combined = list(heuristics)
        combined.extend(_apply_custom_rules(diff_str, "config", review_config))
        real_tools_used: list[str] = []
        config_details: dict[str, Any] = {}

        if any(item in result["file_types"] for item in {"yaml", "github_actions"}):
            yaml_details, yaml_findings = _yaml_findings(targets)
            config_details["yaml"] = yaml_details
            combined.extend(yaml_findings)
            if yaml_details["parsed_files"] or yaml_details["errors"]:
                real_tools_used.append("pyyaml")

        if "dockerfile" in result["file_types"]:
            docker_details, docker_findings = _dockerfile_findings(targets)
            config_details["dockerfile"] = docker_details
            combined.extend(docker_findings)
            if docker_details["inspected_files"]:
                real_tools_used.append("dockerfile_fixture_scan")

        result.update(
            {
                "analysis_mode": targets["analysis_mode"],
                "score": _score_from_findings(combined, penalty=0.30),
                "findings": combined,
                "config_tooling": config_details,
                "real_tools_used": real_tools_used,
            }
        )
    return result


@mcp.tool
def submit_review(verdict: str, reasoning: str, confidence: float | None = None) -> dict[str, Any]:
    """Submit the final PR review verdict."""
    normalized = verdict.strip().lower()
    if normalized not in {"approve", "request_changes", "reject"}:
        raise ValueError("verdict must be approve, request_changes, or reject")
    normalized_confidence = None
    if confidence is not None:
        normalized_confidence = max(0.0, min(1.0, float(confidence)))
    return {
        "terminal": True,
        "verdict": normalized,
        "confidence": normalized_confidence,
        "reasoning": reasoning.strip(),
    }


@mcp.tool
def escalate(reason: str) -> dict[str, Any]:
    """Escalate the PR to a human reviewer."""
    return {
        "terminal": True,
        "verdict": "escalate",
        "reasoning": reason.strip(),
    }


TOOL_REGISTRY = {
    "check_security": check_security,
    "check_quality": check_quality,
    "check_build_and_types": check_build_and_types,
    "check_tests": check_tests,
    "check_config": check_config,
    "submit_review": submit_review,
    "escalate": escalate,
}
