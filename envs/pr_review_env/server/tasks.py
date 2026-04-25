from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).parent.parent.parent.parent
SEED_TASKS_PATH = REPO_ROOT / "tasks" / "tasks.jsonl"
COMPREHENSIVE_TASKS_PATH = REPO_ROOT / "tasks" / "comprehensive_tasks.jsonl"
ALL_TASKS_PATH = REPO_ROOT / "tasks" / "all_tasks.jsonl"
TASKS_PATH = ALL_TASKS_PATH


@dataclass
class PRTask:
    task_id: str
    diff_str: str
    pr_description: str
    primary_language: str
    changed_file_types: list[str]
    repo_kind: str
    expected_verdict: str
    risk_domains: list[str]
    difficulty: str
    review_goal: str
    ownership_hint: str = ""
    author_level: str = "mid"       # junior | mid | senior | lead | non_tech | unknown
    notes: list[str] = field(default_factory=list)
    context_requirements: list[str] = field(default_factory=list)
    expected_evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


_CONTEXT_METADATA_BY_TASK_ID: dict[str, dict[str, list[str]]] = {
    "comp_auth_jwt_removed": {
        "context_requirements": ["auth_middleware_must_verify_jwt_signature"],
        "expected_evidence": ["architecture_summary", "check_security", "custom_rules"],
    },
    "comp_py_logging_pii": {
        "context_requirements": ["auth_flows_must_not_log_credentials_or_tokens"],
        "expected_evidence": ["architecture_summary", "check_security", "custom_rules"],
    },
    "comp_django_irreversible_migration": {
        "context_requirements": ["migrations_require_rollback_plan"],
        "expected_evidence": ["docs/features.md", "check_quality", "check_build_and_types"],
    },
    "comp_k8s_resource_limits_removed": {
        "context_requirements": ["infrastructure_must_preserve_limits_and_health_probes"],
        "expected_evidence": ["architecture_summary", "check_config", "custom_rules"],
    },
    "comp_py_rate_limit_removed": {
        "context_requirements": ["api_rate_limiting_must_remain_enforced"],
        "expected_evidence": ["docs/features.md", "check_security", "check_config"],
    },
    "docker_root_user": {
        "context_requirements": ["containers_require_non_root_runtime"],
        "expected_evidence": ["check_config", "custom_rules"],
    },
    "gha_permissions_write": {
        "context_requirements": ["github_actions_permissions_must_be_least_privilege"],
        "expected_evidence": ["check_config", "custom_rules"],
    },
}


def _apply_context_metadata(task: PRTask) -> PRTask:
    metadata = _CONTEXT_METADATA_BY_TASK_ID.get(task.task_id)
    if not metadata:
        return task
    if not task.context_requirements:
        task.context_requirements = list(metadata["context_requirements"])
    if not task.expected_evidence:
        task.expected_evidence = list(metadata["expected_evidence"])
    return task


def _fallback_tasks() -> list[PRTask]:
    return [
        PRTask(
            task_id="py_sql_injection",
            pr_description="Adds a Flask search endpoint for customer records.",
            diff_str="""diff --git a/app/search.py b/app/search.py
index 1111111..2222222 100644
--- a/app/search.py
+++ b/app/search.py
@@
+from flask import request
+
+def search_users(conn):
+    query = request.args.get("q", "")
+    sql = f"SELECT * FROM users WHERE name LIKE '%{query}%'"
+    return conn.execute(sql).fetchall()
""",
            primary_language="python",
            changed_file_types=["python"],
            repo_kind="backend_service",
            expected_verdict="reject",
            risk_domains=["security"],
            difficulty="easy",
            review_goal="Catch obvious injection and avoid approving without evidence.",
            ownership_hint="backend",
        ),
        PRTask(
            task_id="ts_eval_template",
            pr_description="Adds runtime expression support to a templating helper.",
            diff_str="""diff --git a/web/template.ts b/web/template.ts
index 1111111..2222222 100644
--- a/web/template.ts
+++ b/web/template.ts
@@
+export function renderExpression(input: string) {
+  return eval(input);
+}
""",
            primary_language="typescript",
            changed_file_types=["typescript"],
            repo_kind="frontend_service",
            expected_verdict="reject",
            risk_domains=["security"],
            difficulty="easy",
            review_goal="Reject direct code execution in user-facing code.",
            ownership_hint="frontend",
        ),
        PRTask(
            task_id="java_null_contract",
            pr_description="Simplifies null handling in a service layer helper.",
            diff_str="""diff --git a/src/main/java/com/acme/UserService.java b/src/main/java/com/acme/UserService.java
index 1111111..2222222 100644
--- a/src/main/java/com/acme/UserService.java
+++ b/src/main/java/com/acme/UserService.java
@@
+public String fullName(User user) {
+    return user.getProfile().getDisplayName().trim();
+}
""",
            primary_language="java",
            changed_file_types=["java"],
            repo_kind="backend_service",
            expected_verdict="request_changes",
            risk_domains=["quality"],
            difficulty="medium",
            review_goal="Notice missing null safety and brittle contracts.",
            ownership_hint="platform",
        ),
        PRTask(
            task_id="go_clean_refactor",
            pr_description="Extracts request validation into a helper in the Go API.",
            diff_str="""diff --git a/api/validate.go b/api/validate.go
index 1111111..2222222 100644
--- a/api/validate.go
+++ b/api/validate.go
@@
+func validateID(id string) error {
+    if id == "" {
+        return errors.New("missing id")
+    }
+    return nil
+}
""",
            primary_language="go",
            changed_file_types=["go"],
            repo_kind="backend_service",
            expected_verdict="approve",
            risk_domains=[],
            difficulty="easy",
            review_goal="Avoid over-calling tools on straightforward clean diffs.",
            ownership_hint="backend",
        ),
        PRTask(
            task_id="rust_unwrap_io",
            pr_description="Adds a helper to load cached configuration in Rust.",
            diff_str="""diff --git a/src/cache.rs b/src/cache.rs
index 1111111..2222222 100644
--- a/src/cache.rs
+++ b/src/cache.rs
@@
+pub fn load_cache(path: &str) -> String {
+    std::fs::read_to_string(path).unwrap()
+}
""",
            primary_language="rust",
            changed_file_types=["rust"],
            repo_kind="cli_tool",
            expected_verdict="request_changes",
            risk_domains=["quality"],
            difficulty="easy",
            review_goal="Flag panic-prone error handling in runtime code.",
            ownership_hint="systems",
        ),
        PRTask(
            task_id="docker_root_user",
            pr_description="Adds a Dockerfile for the review worker service.",
            diff_str="""diff --git a/Dockerfile b/Dockerfile
new file mode 100644
--- /dev/null
+++ b/Dockerfile
@@
+FROM python:3.12-slim
+WORKDIR /app
+COPY . .
+RUN pip install -r requirements.txt
+CMD ["python", "worker.py"]
""",
            primary_language="python",
            changed_file_types=["dockerfile"],
            repo_kind="backend_service",
            expected_verdict="request_changes",
            risk_domains=["config", "security"],
            difficulty="medium",
            review_goal="Inspect container posture, not just source code.",
            ownership_hint="devops",
        ),
        PRTask(
            task_id="gha_permissions_write",
            pr_description="Introduces a GitHub Actions workflow to run preview deployments.",
            diff_str="""diff --git a/.github/workflows/preview.yml b/.github/workflows/preview.yml
new file mode 100644
--- /dev/null
+++ b/.github/workflows/preview.yml
@@
+name: preview
+on: [pull_request]
+jobs:
+  deploy:
+    permissions:
+      contents: write
+      pull-requests: write
+    runs-on: ubuntu-latest
+    steps:
+      - uses: actions/checkout@v4
+      - run: ./deploy-preview.sh
""",
            primary_language="yaml",
            changed_file_types=["yaml", "github_actions"],
            repo_kind="monorepo",
            expected_verdict="request_changes",
            risk_domains=["config", "security", "build"],
            difficulty="medium",
            review_goal="Catch over-broad workflow permissions in repo-global files.",
            ownership_hint="devops",
        ),
        PRTask(
            task_id="gitignore_secret_file",
            pr_description="Narrows env file ignore rules to only local dev files.",
            diff_str="""diff --git a/.gitignore b/.gitignore
index 1111111..2222222 100644
--- a/.gitignore
+++ b/.gitignore
@@
-.env*
+.env.local
+.env.development
""",
            primary_language="yaml",
            changed_file_types=["gitignore"],
            repo_kind="backend_service",
            expected_verdict="request_changes",
            risk_domains=["config", "security"],
            difficulty="easy",
            review_goal="Catch removal of broad env-file protection: .env.production and .env.staging are no longer ignored after this change.",
            ownership_hint="security",
        ),
        PRTask(
            task_id="yaml_prod_debug",
            pr_description="Enables debug logging in the Kubernetes production deployment.",
            diff_str="""diff --git a/deploy/prod.yaml b/deploy/prod.yaml
index 1111111..2222222 100644
--- a/deploy/prod.yaml
+++ b/deploy/prod.yaml
@@
+spec:
+  template:
+    spec:
+      containers:
+        - name: api
+          env:
+            - name: LOG_LEVEL
+              value: debug
""",
            primary_language="yaml",
            changed_file_types=["yaml"],
            repo_kind="backend_service",
            expected_verdict="request_changes",
            risk_domains=["config"],
            difficulty="easy",
            review_goal="Catch environment-level regressions in infra manifests.",
            ownership_hint="devops",
        ),
        PRTask(
            task_id="mixed_python_workflow",
            pr_description="Adds a Python release script and updates the release workflow.",
            diff_str="""diff --git a/scripts/release.py b/scripts/release.py
index 1111111..2222222 100644
--- a/scripts/release.py
+++ b/scripts/release.py
@@
+def build_tag(version):
+    return f"release-{version}"
diff --git a/.github/workflows/release.yml b/.github/workflows/release.yml
index 1111111..2222222 100644
--- a/.github/workflows/release.yml
+++ b/.github/workflows/release.yml
@@
+      - run: python scripts/release.py
""",
            primary_language="python",
            changed_file_types=["python", "yaml", "github_actions"],
            repo_kind="library",
            expected_verdict="approve",
            risk_domains=["build"],
            difficulty="easy",
            review_goal="Mixed-file PRs are not automatically risky.",
            ownership_hint="release",
        ),
        PRTask(
            task_id="ts_missing_test",
            pr_description="Adds client-side authentication redirect handling.",
            diff_str="""diff --git a/web/auth.ts b/web/auth.ts
index 1111111..2222222 100644
--- a/web/auth.ts
+++ b/web/auth.ts
@@
+export async function redirectToLogin(next: string) {
+  window.location.assign(`/login?next=${encodeURIComponent(next)}`);
+}
""",
            primary_language="typescript",
            changed_file_types=["typescript"],
            repo_kind="frontend_service",
            expected_verdict="request_changes",
            risk_domains=["quality", "tests"],
            difficulty="medium",
            review_goal="Reward asking for tests on behavior changes without rejecting as insecure.",
            ownership_hint="frontend",
        ),
        PRTask(
            task_id="java_dependency_bump",
            pr_description="Bumps a patch dependency version in Maven config.",
            diff_str="""diff --git a/pom.xml b/pom.xml
index 1111111..2222222 100644
--- a/pom.xml
+++ b/pom.xml
@@
+    <dependency>
+      <groupId>org.slf4j</groupId>
+      <artifactId>slf4j-api</artifactId>
+      <version>2.0.16</version>
+    </dependency>
""",
            primary_language="java",
            changed_file_types=["java", "build_manifest"],
            repo_kind="backend_service",
            expected_verdict="approve",
            risk_domains=["build"],
            difficulty="easy",
            review_goal="Do not over-penalize routine dependency maintenance.",
            ownership_hint="platform",
        ),
    ]


def resolve_task_path(path: str | Path | None = None) -> Path:
    if path is None:
        return TASKS_PATH
    value = str(path)
    aliases = {
        "all": ALL_TASKS_PATH,
        "default": ALL_TASKS_PATH,
        "seed": SEED_TASKS_PATH,
        "base": SEED_TASKS_PATH,
        "comprehensive": COMPREHENSIVE_TASKS_PATH,
        "stress": COMPREHENSIVE_TASKS_PATH,
    }
    return aliases.get(value, Path(value))


def _load_jsonl_tasks(path: str | Path) -> list[PRTask]:
    task_path = Path(path)
    if not task_path.exists():
        return [_apply_context_metadata(task) for task in _fallback_tasks()]

    tasks: list[PRTask] = []
    for line in task_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        tasks.append(_apply_context_metadata(PRTask(**payload)))

    return tasks or [_apply_context_metadata(task) for task in _fallback_tasks()]


def _dedupe_tasks(tasks: Iterable[PRTask]) -> list[PRTask]:
    seen: set[str] = set()
    deduped: list[PRTask] = []
    for task in tasks:
        if task.task_id in seen:
            continue
        seen.add(task.task_id)
        deduped.append(task)
    return deduped


def load_tasks(path: str | Path | None = None) -> list[PRTask]:
    task_path = resolve_task_path(path)
    if task_path == ALL_TASKS_PATH and not task_path.exists():
        return _dedupe_tasks([
            *_load_jsonl_tasks(SEED_TASKS_PATH),
            *_load_jsonl_tasks(COMPREHENSIVE_TASKS_PATH),
        ])
    return _load_jsonl_tasks(task_path)


def get_random_task(
    path: str | Path | None = None,
    rng: random.Random | None = None,
) -> PRTask:
    generator = rng or random
    return generator.choice(load_tasks(path))


def get_task_by_id(task_id: str, path: str | Path | None = None) -> PRTask:
    for task in load_tasks(path):
        if task.task_id == task_id:
            return task
    raise KeyError(f"Unknown task_id: {task_id}")


def dump_tasks(tasks: Iterable[PRTask], path: str | Path = TASKS_PATH) -> None:
    lines = [json.dumps(task.to_dict(), ensure_ascii=True) for task in tasks]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def dump_all_tasks(path: str | Path = ALL_TASKS_PATH) -> None:
    dump_tasks(
        _dedupe_tasks([
            *_load_jsonl_tasks(SEED_TASKS_PATH),
            *_load_jsonl_tasks(COMPREHENSIVE_TASKS_PATH),
        ]),
        path,
    )


def _paths_touched(task: PRTask) -> list[str]:
    import re

    return [
        new if new != "/dev/null" else old
        for old, new in re.findall(r"^diff --git a/(.*?) b/(.*?)$", task.diff_str, flags=re.MULTILINE)
    ]


@lru_cache(maxsize=8)
def _structural_config_for_docs(docs_dir: str) -> dict[str, Any]:
    from .context_loader import extract_structural_config

    return extract_structural_config(docs_dir)


_LANG_TOOLS: dict[str, set[str]] = {
    "python":     {"semgrep", "ruff", "pylint", "radon"},
    "typescript": {"semgrep", "tsc", "eslint", "pyright"},
    "javascript": {"semgrep", "eslint", "tsc"},
    "java":       {"semgrep", "javac"},
    "go":         {"semgrep", "go"},
    "rust":       {"semgrep", "cargo"},
    "yaml":       {"pyyaml", "semgrep"},
}
_FILE_TYPE_TOOLS: dict[str, set[str]] = {
    "dockerfile":    {"semgrep"},
    "yaml":          {"pyyaml"},
    "github_actions":{"pyyaml"},
    "build_manifest":{"semgrep"},
}

_REPO_BLURBS: dict[str, str] = {
    "backend_service":  "Backend service. Correctness, security, and observability are priorities.",
    "frontend_service": "Frontend web app. UX correctness, accessibility, and bundle size matter.",
    "cli_tool":         "CLI tool. Robust error handling and clean exit semantics matter; no servers.",
    "library":          "Reusable library. Public API stability, semver, and zero side effects matter.",
    "monorepo":         "Monorepo with shared infrastructure and CI workflows.",
    "custom":           "Single-purpose change.",
}


def _task_relevant_tools(task: PRTask, all_enabled: list[str]) -> list[str]:
    relevant: set[str] = set()
    relevant |= _LANG_TOOLS.get(task.primary_language, set())
    for ftype in task.changed_file_types:
        relevant |= _FILE_TYPE_TOOLS.get(ftype, set())
    if not relevant:
        return list(all_enabled)[:6]
    return [tool for tool in all_enabled if tool in relevant][:6]


def task_review_config(
    task: PRTask,
    docs_dir: str | Path = REPO_ROOT / "docs",
    mode: str = "short",
) -> dict[str, Any] | None:
    """Return a cheap loader-derived config scoped to one task.

    `empty` disables prompt context while still allowing callers to exercise
    the loader path. `short` is intended for training speed. `full` preserves
    the full structural config for app demos and loader analysis.
    """
    if mode in {"", "none", "off"}:
        return None

    config = _structural_config_for_docs(str(docs_dir))
    config = json.loads(json.dumps(config))
    config["extraction_method"] = f"task_{mode}_structural"

    if mode == "empty":
        config["architecture_summary"] = ""
        config["critical_paths"] = []
        config["custom_rules"] = []
        config["enabled_tools"] = []
        config["planned_tools"] = []
        return config

    if mode == "short":
        paths = _paths_touched(task)
        critical = []
        for critical_path in config.get("critical_paths", []):
            normalized = critical_path.rstrip("/")
            if any(path == normalized or path.startswith(f"{normalized}/") for path in paths):
                critical.append(critical_path)
        config["critical_paths"] = critical[:4]
        config["architecture_summary"] = _REPO_BLURBS.get(task.repo_kind, _REPO_BLURBS["custom"])
        config["domain_priorities"] = {
            key: value
            for key, value in config.get("domain_priorities", {}).items()
            if key in set(task.risk_domains)
        }
        risk_domains = set(task.risk_domains)
        kept_rules = []
        for rule in config.get("custom_rules", []):
            domain = str(rule.get("domain", "")).strip().lower()
            pattern = str(rule.get("pattern", ""))
            keep_for_domain = bool(domain and domain in risk_domains)
            keep_for_paths = any(path.startswith(pattern.split("/")[0]) for path in paths if pattern)
            if keep_for_domain or keep_for_paths:
                kept_rules.append(rule)
        config["custom_rules"] = kept_rules[:5]
        config["enabled_tools"] = _task_relevant_tools(task, config.get("enabled_tools", []))
        config["planned_tools"] = []

    return config
