from envs.pr_review_env.server.context_loader import extract_structural_config


def test_extract_structural_config_reads_review_tool_doc():
    config = extract_structural_config("docs")

    assert config["tool_weights"]["check_security"] == 0.40
    assert config["domain_priorities"]["security"] == 1.6
    assert "src/auth/" in config["critical_paths"]
    assert config["author_depth"]["junior"] == 1.5
    assert "semgrep" in config["enabled_tools"]
    assert config["architecture_summary"]
