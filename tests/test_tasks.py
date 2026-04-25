from envs.pr_review_env.server.tasks import get_task_by_id, load_tasks, task_review_config


def test_load_tasks_returns_seed_bank():
    tasks = load_tasks()
    assert tasks
    assert len(tasks) == 78
    assert any(task.primary_language == "python" for task in tasks)
    assert any("dockerfile" in task.changed_file_types for task in tasks)
    assert any(task.primary_language == "java" for task in tasks)
    assert any(task.primary_language == "javascript" for task in tasks)


def test_get_task_by_id_finds_known_seed():
    task = get_task_by_id("py_sql_injection")
    assert task.expected_verdict == "reject"


def test_load_task_aliases_expose_seed_and_comprehensive_banks():
    assert len(load_tasks("seed")) == 65
    assert len(load_tasks("comprehensive")) == 13
    assert len(load_tasks("all")) == 78


def test_task_review_config_short_keeps_loader_context_small():
    task = get_task_by_id("comp_auth_jwt_removed")
    config = task_review_config(task, mode="short")

    assert config is not None
    assert config["extraction_method"] == "task_short_structural"
    assert len(config["architecture_summary"]) <= 240
    assert config["planned_tools"] == []
    assert task.context_requirements
    assert task.expected_evidence
    assert config["custom_rules"]


def test_task_review_config_can_be_empty_or_off():
    task = get_task_by_id("py_sql_injection")
    empty = task_review_config(task, mode="empty")

    assert empty is not None
    assert empty["architecture_summary"] == ""
    assert task_review_config(task, mode="off") is None
