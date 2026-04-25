from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.pr_review_env.server.tasks import PRTask, dump_all_tasks, dump_tasks


def task(
    task_id: str,
    pr_description: str,
    diff_str: str,
    primary_language: str,
    changed_file_types: list[str],
    repo_kind: str,
    expected_verdict: str,
    risk_domains: list[str],
    difficulty: str,
    review_goal: str,
    ownership_hint: str,
) -> PRTask:
    return PRTask(
        task_id=task_id,
        pr_description=pr_description,
        diff_str=diff_str,
        primary_language=primary_language,
        changed_file_types=changed_file_types,
        repo_kind=repo_kind,
        expected_verdict=expected_verdict,
        risk_domains=risk_domains,
        difficulty=difficulty,
        review_goal=review_goal,
        ownership_hint=ownership_hint,
    )


def build_task_bank() -> list[PRTask]:
    tasks: list[PRTask] = []

    tasks.extend(
        [
            task(
                "py_sql_injection",
                "Adds a Flask search endpoint for customer records.",
                """diff --git a/app/search.py b/app/search.py
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
                "python",
                ["python"],
                "backend_service",
                "reject",
                ["security"],
                "easy",
                "Catch obvious injection and avoid approving without evidence.",
                "backend",
            ),
            task(
                "ts_eval_template",
                "Adds runtime expression support to a templating helper.",
                """diff --git a/web/template.ts b/web/template.ts
index 1111111..2222222 100644
--- a/web/template.ts
+++ b/web/template.ts
@@
+export function renderExpression(input: string) {
+  return eval(input);
+}
""",
                "typescript",
                ["typescript"],
                "frontend_service",
                "reject",
                ["security"],
                "easy",
                "Reject direct code execution in user-facing code.",
                "frontend",
            ),
            task(
                "go_clean_refactor",
                "Extracts request validation into a helper in the Go API.",
                """diff --git a/api/validate.go b/api/validate.go
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
                "go",
                ["go"],
                "backend_service",
                "approve",
                ["clean"],
                "easy",
                "Avoid over-calling tools on straightforward clean diffs.",
                "backend",
            ),
            task(
                "rust_unwrap_io",
                "Adds a helper to load cached configuration in Rust.",
                """diff --git a/src/cache.rs b/src/cache.rs
index 1111111..2222222 100644
--- a/src/cache.rs
+++ b/src/cache.rs
@@
+pub fn load_cache(path: &str) -> String {
+    std::fs::read_to_string(path).unwrap()
+}
""",
                "rust",
                ["rust"],
                "cli_tool",
                "request_changes",
                ["quality"],
                "easy",
                "Flag panic-prone error handling in runtime code.",
                "systems",
            ),
            task(
                "docker_root_user",
                "Adds a Dockerfile for the review worker service.",
                """diff --git a/Dockerfile b/Dockerfile
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
                "python",
                ["dockerfile"],
                "backend_service",
                "request_changes",
                ["config", "security"],
                "medium",
                "Inspect container posture, not just source code.",
                "devops",
            ),
            task(
                "gha_permissions_write",
                "Introduces a GitHub Actions workflow to run preview deployments.",
                """diff --git a/.github/workflows/preview.yml b/.github/workflows/preview.yml
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
                "yaml",
                ["yaml", "github_actions"],
                "monorepo",
                "request_changes",
                ["config", "security", "ci"],
                "medium",
                "Catch over-broad workflow permissions in repo-global files.",
                "devops",
            ),
            task(
                "gitignore_secret_file",
                "Updates ignore rules after adding a local env file.",
                """diff --git a/.gitignore b/.gitignore
index 1111111..2222222 100644
--- a/.gitignore
+++ b/.gitignore
@@
+secrets/
+.env.production
""",
                "yaml",
                ["gitignore"],
                "backend_service",
                "request_changes",
                ["config", "security"],
                "easy",
                "Review ignore patterns that can hide leaked secrets or deploy config.",
                "security",
            ),
            task(
                "mixed_python_workflow",
                "Adds a Python release script and updates the release workflow.",
                """diff --git a/scripts/release.py b/scripts/release.py
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
                "python",
                ["python", "yaml", "github_actions"],
                "library",
                "approve",
                ["clean", "ci"],
                "easy",
                "Mixed-file PRs are not automatically risky.",
                "release",
            ),
        ]
    )

    security_cases = [
        (
            "python",
            "backend_service",
            "py_pickle_deser",
            "Adds a cache loader for user-submitted session files.",
            """diff --git a/app/session_cache.py b/app/session_cache.py
@@
+import pickle
+
+def load_session(blob: bytes):
+    return pickle.loads(blob)
""",
            ["python"],
            "Reject unsafe deserialization paths.",
            "backend",
        ),
        (
            "javascript",
            "frontend_service",
            "js_innerhtml_sink",
            "Adds a DOM helper for rendering announcement content.",
            """diff --git a/web/banner.js b/web/banner.js
@@
+export function showBanner(node, html) {
+  node.innerHTML = html;
+}
""",
            ["javascript"],
            "Catch obvious client-side injection sinks.",
            "frontend",
        ),
        (
            "java",
            "backend_service",
            "java_runtime_exec",
            "Adds a helper for invoking a local maintenance script.",
            """diff --git a/src/main/java/com/acme/Maintenance.java b/src/main/java/com/acme/Maintenance.java
@@
+public void runScript(String cmd) throws Exception {
+    Runtime.getRuntime().exec(cmd);
+}
""",
            ["java"],
            "Reject shell command execution from request-controlled paths.",
            "platform",
        ),
        (
            "go",
            "backend_service",
            "go_hardcoded_token",
            "Adds a quick integration helper for an internal API.",
            """diff --git a/internal/integration/client.go b/internal/integration/client.go
@@
+const apiToken = "prod-token-123"
+
+func authHeader() string {
+    return "Bearer " + apiToken
+}
""",
            ["go"],
            "Catch hardcoded credentials in production code.",
            "backend",
        ),
        (
            "rust",
            "cli_tool",
            "rust_command_exec",
            "Adds a diagnostics helper to run a shell command.",
            """diff --git a/src/diag.rs b/src/diag.rs
@@
+pub fn run_diag(input: &str) {
+    std::process::Command::new("sh").arg("-c").arg(input).status().unwrap();
+}
""",
            ["rust"],
            "Reject unsafe shell execution in diagnostics helpers.",
            "systems",
        ),
    ]
    for language, repo_kind, task_id, description, diff, file_types, goal, owner in security_cases:
        tasks.append(
            task(
                task_id,
                description,
                diff,
                language,
                file_types,
                repo_kind,
                "reject",
                ["security"],
                "medium",
                goal,
                owner,
            )
        )

    quality_cases = [
        (
            "python",
            "py_nested_branching",
            "Introduces feature-flag branching into a Python request handler.",
            """diff --git a/app/handler.py b/app/handler.py
@@
+def handle(req, flags):
+    if flags.a:
+        if req.user:
+            if req.user.is_admin:
+                return "admin-a"
+            elif req.user.is_staff:
+                return "staff-a"
+        else:
+            return "anon-a"
+    else:
+        if req.user and req.user.is_admin:
+            return "admin-b"
+    return "default"
""",
            ["python"],
        ),
        (
            "typescript",
            "ts_missing_test_redirect",
            "Adds a browser redirect helper for auth fallback.",
            """diff --git a/web/auth.ts b/web/auth.ts
@@
+export async function redirectToLogin(next: string) {
+  window.location.assign(`/login?next=${encodeURIComponent(next)}`);
+}
""",
            ["typescript"],
        ),
        (
            "java",
            "java_null_contract",
            "Simplifies null handling in a service layer helper.",
            """diff --git a/src/main/java/com/acme/UserService.java b/src/main/java/com/acme/UserService.java
@@
+public String fullName(User user) {
+    return user.getProfile().getDisplayName().trim();
+}
""",
            ["java"],
        ),
        (
            "go",
            "go_missing_error_wrap",
            "Adds a parser helper for loading config files.",
            """diff --git a/config/load.go b/config/load.go
@@
+func Load(path string) error {
+    _, err := os.ReadFile(path)
+    if err != nil {
+        return err
+    }
+    return nil
+}
""",
            ["go"],
        ),
        (
            "rust",
            "rust_todo_prod",
            "Adds a temporary parser for migration metadata.",
            """diff --git a/src/migrate.rs b/src/migrate.rs
@@
+pub fn parse_meta(raw: &str) -> &str {
+    todo!()
+}
""",
            ["rust"],
        ),
    ]
    for language, task_id, description, diff, file_types in quality_cases:
        tasks.append(
            task(
                task_id,
                description,
                diff,
                language,
                file_types,
                "backend_service" if language != "typescript" else "frontend_service",
                "request_changes",
                ["quality", "tests"],
                "medium",
                "Push reviewers toward quality or test evidence instead of straight rejection.",
                "platform" if language in {"java", "rust"} else "backend",
            )
        )

    clean_cases = [
        (
            "python",
            "py_helper_extract",
            "Extracts validation into a small helper in the Python API.",
            """diff --git a/app/validate.py b/app/validate.py
@@
+def ensure_non_empty(value: str) -> bool:
+    return bool(value and value.strip())
""",
            ["python"],
        ),
        (
            "typescript",
            "ts_dependency_bump",
            "Bumps a patch dependency in the frontend package manifest.",
            """diff --git a/package.json b/package.json
@@
+    "react-router-dom": "7.1.2"
""",
            ["typescript", "build_manifest"],
        ),
        (
            "java",
            "java_dependency_bump",
            "Bumps a patch dependency version in Maven config.",
            """diff --git a/pom.xml b/pom.xml
@@
+    <dependency>
+      <groupId>org.slf4j</groupId>
+      <artifactId>slf4j-api</artifactId>
+      <version>2.0.16</version>
+    </dependency>
""",
            ["java", "build_manifest"],
        ),
        (
            "go",
            "go_logging_refactor",
            "Extracts logging prefix generation in a Go worker.",
            """diff --git a/worker/logging.go b/worker/logging.go
@@
+func prefix(queue string) string {
+    return "[worker:" + queue + "]"
+}
""",
            ["go"],
        ),
        (
            "rust",
            "rust_error_enum_add",
            "Adds a typed error enum variant for a new parsing case.",
            """diff --git a/src/error.rs b/src/error.rs
@@
+    #[error("invalid metadata version")]
+    InvalidMetadataVersion,
""",
            ["rust"],
        ),
    ]
    for language, task_id, description, diff, file_types in clean_cases:
        tasks.append(
            task(
                task_id,
                description,
                diff,
                language,
                file_types,
                "library" if language in {"rust", "java"} else "backend_service",
                "approve",
                ["clean", "build"] if "build_manifest" in file_types else ["clean"],
                "easy",
                "Avoid wasting steps on routine safe changes.",
                "release" if "build_manifest" in file_types else "backend",
            )
        )

    config_cases = [
        (
            "yaml_prod_debug",
            "Enables debug logging in the Kubernetes production deployment.",
            """diff --git a/deploy/prod.yaml b/deploy/prod.yaml
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
            "request_changes",
            ["yaml"],
            ["config"],
        ),
        (
            "docker_latest_node",
            "Adds a Node service Dockerfile using a broad base image.",
            """diff --git a/Dockerfile b/Dockerfile
@@
+FROM node:latest
+WORKDIR /srv
+COPY . .
+CMD ["node", "server.js"]
""",
            "request_changes",
            ["dockerfile"],
            ["config", "security"],
        ),
        (
            "gha_release_token",
            "Adds a release workflow that writes repository contents.",
            """diff --git a/.github/workflows/release.yml b/.github/workflows/release.yml
@@
+permissions:
+  contents: write
+  actions: write
+jobs:
+  release:
+    runs-on: ubuntu-latest
+    steps:
+      - uses: actions/checkout@v4
+      - run: ./release.sh
""",
            "request_changes",
            ["yaml", "github_actions"],
            ["config", "security", "ci"],
        ),
        (
            "gitignore_dist_only",
            "Adds build output to ignore rules.",
            """diff --git a/.gitignore b/.gitignore
@@
+dist/
+coverage/
""",
            "approve",
            ["gitignore"],
            ["clean"],
        ),
        (
            "yaml_cpu_limit",
            "Adds CPU limits to the worker deployment manifest.",
            """diff --git a/deploy/worker.yaml b/deploy/worker.yaml
@@
+resources:
+  requests:
+    cpu: \"250m\"
+  limits:
+    cpu: \"500m\"
""",
            "approve",
            ["yaml"],
            ["clean", "config"],
        ),
    ]
    for task_id, description, diff, verdict, file_types, risks in config_cases:
        tasks.append(
            task(
                task_id,
                description,
                diff,
                "yaml",
                file_types,
                "monorepo",
                verdict,
                risks,
                "medium" if verdict != "approve" else "easy",
                "Treat repo-global file changes as first-class review cases.",
                "devops",
            )
        )

    mixed_cases = [
        (
            "py_plus_docker_secret",
            "Adds a Python cache helper and a Docker build arg for a token.",
            """diff --git a/app/cache.py b/app/cache.py
@@
+def cache_key(user_id: str) -> str:
+    return f"user:{user_id}"
diff --git a/Dockerfile b/Dockerfile
@@
+ARG API_TOKEN=hardcoded-token
+ENV API_TOKEN=$API_TOKEN
""",
            "reject",
            ["python", "dockerfile"],
            ["security", "config"],
        ),
        (
            "ts_plus_workflow_preview",
            "Adds a frontend preview helper and a preview workflow.",
            """diff --git a/web/preview.ts b/web/preview.ts
@@
+export function previewPath(slug: string) {
+  return `/preview/${slug}`;
+}
diff --git a/.github/workflows/preview.yml b/.github/workflows/preview.yml
@@
+jobs:
+  preview:
+    permissions:
+      contents: read
+    steps:
+      - uses: actions/checkout@v4
+      - run: npm run preview
""",
            "approve",
            ["typescript", "yaml", "github_actions"],
            ["clean", "ci"],
        ),
        (
            "java_plus_manifest",
            "Adds a Java helper and updates Maven build config.",
            """diff --git a/src/main/java/com/acme/Slug.java b/src/main/java/com/acme/Slug.java
@@
+public String normalize(String value) {
+    return value.trim().toLowerCase();
+}
diff --git a/pom.xml b/pom.xml
@@
+    <maven.compiler.release>21</maven.compiler.release>
""",
            "request_changes",
            ["java", "build_manifest"],
            ["quality", "build"],
        ),
        (
            "go_plus_yaml_rollout",
            "Adds a Go rollout helper and updates deployment rollout config.",
            """diff --git a/internal/rollout/state.go b/internal/rollout/state.go
@@
+func Stable() string {
+    return "stable"
+}
diff --git a/deploy/prod.yaml b/deploy/prod.yaml
@@
+strategy:
+  rollingUpdate:
+    maxUnavailable: 0
""",
            "approve",
            ["go", "yaml"],
            ["clean", "config"],
        ),
        (
            "rust_plus_gitignore_target",
            "Adds a Rust parser helper and ignores build output.",
            """diff --git a/src/parse.rs b/src/parse.rs
@@
+pub fn parse_id(raw: &str) -> &str {
+    raw.trim()
+}
diff --git a/.gitignore b/.gitignore
@@
+target/
""",
            "approve",
            ["rust", "gitignore"],
            ["clean"],
        ),
    ]
    for task_id, description, diff, verdict, file_types, risks in mixed_cases:
        primary = next((item for item in file_types if item in {"python", "typescript", "java", "go", "rust"}), "yaml")
        tasks.append(
            task(
                task_id,
                description,
                diff,
                primary,
                file_types,
                "monorepo",
                verdict,
                risks,
                "medium",
                "Mixed PRs should force tool-routing, not one-size-fits-all review.",
                "platform",
            )
        )

    # Create deterministic difficulty variants to reach a hackathon-sized bank.
    variants: list[PRTask] = []
    for original in list(tasks):
        if len(variants) >= 32:
            break
        variant_id = f"{original.task_id}_variant"
        variant_diff = original.diff_str + "\n+// review-note: maintain parity with sibling service\n"
        if "python" in original.changed_file_types:
            variant_diff = original.diff_str + "\n+# review-note: keep behavior aligned with batch worker\n"
        elif "yaml" in original.changed_file_types:
            variant_diff = original.diff_str + "\n+# managed-by: rollout controller\n"

        variants.append(
            PRTask(
                task_id=variant_id,
                diff_str=variant_diff,
                pr_description=f"{original.pr_description} Variant for another service boundary.",
                primary_language=original.primary_language,
                changed_file_types=list(original.changed_file_types),
                repo_kind=original.repo_kind,
                expected_verdict=original.expected_verdict,
                risk_domains=list(original.risk_domains),
                difficulty="hard" if original.difficulty != "hard" else "medium",
                review_goal=original.review_goal,
                ownership_hint=original.ownership_hint,
            )
        )

    tasks.extend(variants)
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the deterministic PR review task bank.")
    parser.add_argument(
        "--output",
        default=str(ROOT / "tasks" / "tasks.jsonl"),
        help="Path to the output JSONL file.",
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print a small JSON summary after writing the file.",
    )
    parser.add_argument(
        "--all-output",
        default=str(ROOT / "tasks" / "all_tasks.jsonl"),
        help="Path to the combined seed+comprehensive JSONL file.",
    )
    parser.add_argument(
        "--skip-all",
        action="store_true",
        help="Only write the seed bank, not the combined all-task bank.",
    )
    args = parser.parse_args()

    task_bank = build_task_bank()
    dump_tasks(task_bank, args.output)
    if not args.skip_all:
        dump_all_tasks(args.all_output)

    if args.print_summary:
        languages: dict[str, int] = {}
        for item in task_bank:
            languages[item.primary_language] = languages.get(item.primary_language, 0) + 1
        print(
            json.dumps(
                {
                    "tasks_written": len(task_bank),
                    "output": str(args.output),
                    "all_output": "" if args.skip_all else str(args.all_output),
                    "by_primary_language": languages,
                }
            )
        )


if __name__ == "__main__":
    main()
