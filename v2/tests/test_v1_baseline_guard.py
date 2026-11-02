"""The smoke test's V1 guard: exact exceptions pass, and tampering fails."""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
GUARD = REPOSITORY / "v2/dev/check_v1_baseline.sh"
TAG = "technical-test-v1.0"
BYTECODE = "pipeline/__pycache__/02_stats.cpython-313.pyc"


def git(root: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "-c", "commit.gpgsign=false", *arguments],
        cwd=root, check=True, capture_output=True, env={**os.environ, "GIT_CONFIG_GLOBAL": os.devnull},
    )


def guard(root: Path, **environment: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(GUARD)], cwd=root, capture_output=True, text=True, env={**os.environ, **environment},
    )


def commit(root: Path) -> None:
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "change")


def write(root: Path, relative: str, content: str = "changed\n") -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


@pytest.fixture
def baseline(tmp_path: Path) -> Path:
    git(tmp_path, "init", "-q")
    for relative in (
        "pipeline/02_stats.py",
        "dashboard/app.py",
        "scripts/gen_dummy_data.py",
        "scripts/ci/tagged.sh",
        "data/processed/meta.json",
        BYTECODE,
        "pipeline/__pycache__/unlisted.cpython-313.pyc",
    ):
        write(tmp_path, relative, "original\n")
    commit(tmp_path)
    git(tmp_path, "tag", TAG)
    return tmp_path


def test_unchanged_baseline_passes(baseline):
    assert guard(baseline).returncode == 0


def test_known_bytecode_removal_and_new_ci_scripts_pass(baseline):
    (baseline / BYTECODE).unlink()
    write(baseline, "scripts/ci/smoke-staging.sh")
    commit(baseline)
    write(baseline, "scripts/ci/untracked-helper.sh")
    assert guard(baseline).returncode == 0


@pytest.mark.parametrize(
    "relative",
    ["pipeline/02_stats.py", "dashboard/app.py", "scripts/gen_dummy_data.py", "data/processed/meta.json"],
)
@pytest.mark.parametrize("committed", [True, False], ids=["committed", "working-tree"])
def test_protected_edits_fail(baseline, relative, committed):
    write(baseline, relative)
    if committed:
        commit(baseline)
    result = guard(baseline)
    assert result.returncode == 1 and relative in result.stderr


def test_protected_deletion_fails(baseline):
    (baseline / "pipeline/02_stats.py").unlink()
    commit(baseline)
    result = guard(baseline)
    assert result.returncode == 1 and "pipeline/02_stats.py" in result.stderr


def test_moving_protected_source_into_the_ci_exception_fails(baseline):
    git(baseline, "mv", "pipeline/02_stats.py", "scripts/ci/02_stats.py")
    commit(baseline)
    result = guard(baseline)
    assert result.returncode == 1 and "pipeline/02_stats.py" in result.stderr


@pytest.mark.parametrize("relative", ["pipeline/injected.py", "dashboard/pages/injected.py"])
@pytest.mark.parametrize("committed", [True, False], ids=["committed", "untracked"])
def test_new_files_in_protected_paths_fail(baseline, relative, committed):
    write(baseline, relative)
    if committed:
        commit(baseline)
    result = guard(baseline)
    assert result.returncode == 1 and relative in result.stderr


def test_exceptions_allow_only_their_own_change(baseline):
    write(baseline, BYTECODE)
    (baseline / "pipeline/__pycache__/unlisted.cpython-313.pyc").unlink()
    write(baseline, "scripts/ci/tagged.sh")
    commit(baseline)
    result = guard(baseline)
    assert result.returncode == 1
    for relative in (BYTECODE, "pipeline/__pycache__/unlisted.cpython-313.pyc", "scripts/ci/tagged.sh"):
        assert relative in result.stderr


def test_guard_checks_the_whole_repository_from_a_subdirectory(baseline):
    write(baseline, "pipeline/02_stats.py")
    assert guard(baseline / "dashboard").returncode == 1


def test_environment_cannot_replace_the_baseline(baseline):
    write(baseline, "pipeline/02_stats.py")
    commit(baseline)
    assert guard(baseline, FOOTBALLAI_V1_BASELINE_REF="HEAD").returncode == 1


@pytest.mark.parametrize("flag", ["--assume-unchanged", "--skip-worktree"])
def test_index_flags_cannot_hide_edits(baseline, flag):
    git(baseline, "update-index", flag, "pipeline/02_stats.py")
    write(baseline, "pipeline/02_stats.py")
    result = guard(baseline)
    assert result.returncode == 1 and "pipeline/02_stats.py" in result.stderr


def test_a_missing_baseline_fails_closed(tmp_path):
    git(tmp_path, "init", "-q")
    result = guard(tmp_path)
    assert result.returncode == 2 and TAG in result.stderr


def test_guard_is_executable():
    assert GUARD.stat().st_mode & stat.S_IXUSR


def test_this_repository_matches_its_declared_exceptions():
    tagged = subprocess.run(["git", "rev-parse", "--verify", "--quiet", f"{TAG}^{{commit}}"], cwd=REPOSITORY)
    if tagged.returncode != 0:
        pytest.skip("the V1 baseline tag is not in this checkout")
    result = guard(REPOSITORY)
    assert result.returncode == 0, result.stderr
