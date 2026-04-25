from envs.pr_review_env.server.tasks import get_task_by_id, load_tasks


def test_load_tasks_returns_seed_bank():
    tasks = load_tasks()
    assert tasks
    assert len(tasks) >= 50
    assert any(task.primary_language == "python" for task in tasks)
    assert any("dockerfile" in task.changed_file_types for task in tasks)
    assert any(task.primary_language == "java" for task in tasks)
    assert any(task.primary_language == "javascript" for task in tasks)


def test_get_task_by_id_finds_known_seed():
    task = get_task_by_id("py_sql_injection")
    assert task.expected_verdict == "reject"
