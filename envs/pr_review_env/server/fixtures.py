from __future__ import annotations

import re
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES_ROOT = REPO_ROOT / "fixtures" / "pr_review"


def changed_paths_from_diff(diff_str: str) -> list[str]:
    paths: list[str] = []
    for old_path, new_path in re.findall(
        r"^diff --git a/(.*?) b/(.*?)$",
        diff_str,
        flags=re.MULTILINE,
    ):
        paths.append(new_path if new_path != "/dev/null" else old_path)
    return paths


def fixture_dir(task_id: str) -> Path:
    return FIXTURES_ROOT / task_id


def has_fixture(task_id: str | None) -> bool:
    return bool(task_id) and fixture_dir(task_id).exists()


@contextmanager
def materialize_workspace(task_id: str) -> Iterator[Path]:
    source = fixture_dir(task_id)
    if not source.exists():
        raise FileNotFoundError(f"No fixture workspace for task_id={task_id}")

    with tempfile.TemporaryDirectory(prefix=f"pr-review-{task_id}-") as tmpdir:
        workspace = Path(tmpdir) / "workspace"
        shutil.copytree(source, workspace)
        yield workspace


def changed_files_in_workspace(workspace: Path, diff_str: str) -> list[Path]:
    files = [workspace / relative for relative in changed_paths_from_diff(diff_str)]
    existing = [path for path in files if path.exists() and path.is_file()]
    if existing:
        return existing

    return [path for path in workspace.rglob("*") if path.is_file()]
