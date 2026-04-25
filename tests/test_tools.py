import shutil

import pytest

from envs.pr_review_env.models import CustomRule, ReviewConfig
from envs.pr_review_env.server.tools import check_build_and_types, check_config, check_security
from envs.pr_review_env.server.tasks import get_task_by_id


def test_check_config_uses_fixture_backed_yaml_parsing():
    task = get_task_by_id("yaml_prod_debug")
    result = check_config(task.diff_str, task_id=task.task_id)
    assert result["analysis_mode"] == "fixture_backed"
    assert "pyyaml" in result["real_tools_used"]


@pytest.mark.skipif(shutil.which("go") is None, reason="go not in PATH")
def test_check_build_and_types_uses_fixture_workspace_for_go(monkeypatch):
    monkeypatch.setenv("PR_REVIEW_TOOL_BACKEND", "hybrid")
    task = get_task_by_id("go_clean_refactor")
    result = check_build_and_types(task.diff_str, task_id=task.task_id)
    assert result["analysis_mode"] == "fixture_backed"
    assert "go" in result["real_tools_used"]


def test_check_security_reports_analysis_mode():
    task = get_task_by_id("py_sql_injection")
    result = check_security(task.diff_str, task_id=task.task_id)
    assert result["analysis_mode"] in {"fixture_backed", "diff_reconstructed"}


def test_check_security_applies_custom_rules():
    task = get_task_by_id("py_sql_injection")
    config = ReviewConfig(
        custom_rules=[
            CustomRule(
                domain="security",
                pattern="SELECT \\*",
                severity="critical",
                message="Raw SELECT star queries are banned.",
            )
        ]
    )

    result = check_security(
        task.diff_str,
        task_id=task.task_id,
        review_config=config.model_dump(),
    )

    assert any("Raw SELECT star" in finding for finding in result["findings"])
