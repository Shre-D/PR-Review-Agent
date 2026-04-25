from envs.pr_review_env.server.tool_registry import (
    TOOL_REGISTRY,
    build_domain_to_tools,
    implemented_tools,
    normalize_findings,
    tools_for_domain,
    tools_for_language,
)


def test_registry_contains_implemented_and_planned_tools():
    assert "semgrep" in implemented_tools()
    assert TOOL_REGISTRY["bandit"].status == "planned"


def test_registry_lookup_helpers_filter_by_domain_and_language():
    security = tools_for_domain("security")
    python_tools = tools_for_language("python")

    assert any(spec.name == "semgrep" for spec in security)
    assert any(spec.name == "ruff" for spec in python_tools)


def test_build_domain_to_tools_maps_specs_to_mcp_tools():
    mapping = build_domain_to_tools()

    assert "check_security" in mapping["security"]
    assert "check_config" in mapping["config"]


def test_normalize_findings_clamps_scores_and_preserves_findings():
    result = normalize_findings(
        "check_security",
        {"score": 1.8, "findings": ["x"], "severity_counts": {"critical": 2}},
    )

    assert result.score == 1.0
    assert result.findings == ("x",)
    assert result.severity_counts["critical"] == 2
