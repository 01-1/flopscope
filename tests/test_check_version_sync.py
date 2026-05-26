"""Tests for scripts/check_version_sync.py."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "check_version_sync.py"


def _run(cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run the version-sync script with cwd, return CompletedProcess."""
    return subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=cwd,
        capture_output=True,
        text=True,
    )


def test_real_repo_is_in_sync():
    """The actual repo at HEAD must be in sync. If this fails, fix the drift."""
    result = _run(REPO_ROOT)
    assert result.returncode == 0, (
        f"check_version_sync failed on the real repo.\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


@pytest.fixture
def repo_copy(tmp_path: Path) -> Path:
    """Copy the version-bearing files into a temp dir mimicking repo structure."""
    dest = tmp_path / "repo"
    dest.mkdir()
    files = [
        "pyproject.toml",
        "src/flopscope/__init__.py",
        "flopscope-server/pyproject.toml",
        "flopscope-server/src/flopscope_server/__init__.py",
        "flopscope-client/pyproject.toml",
        "flopscope-client/src/flopscope/__init__.py",
    ]
    for rel in files:
        src = REPO_ROOT / rel
        dst = dest / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)
    script_dst = dest / "scripts" / "check_version_sync.py"
    script_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(SCRIPT, script_dst)
    return dest


def test_root_pyproject_drift_detected(repo_copy: Path):
    """Bumping root pyproject without bumping subpackages fails the check."""
    pp = repo_copy / "pyproject.toml"
    pp.write_text(pp.read_text().replace('version = "0.3.0"', 'version = "0.3.1"', 1))
    result = _run(repo_copy)
    assert result.returncode != 0
    combined = (result.stdout + result.stderr).lower()
    assert "0.3.1" in combined or "0.3.0" in combined


def test_init_file_drift_detected(repo_copy: Path):
    """Bumping only an __init__ file fails the check."""
    init = repo_copy / "flopscope-server" / "src" / "flopscope_server" / "__init__.py"
    init.write_text(init.read_text().replace('"0.3.0"', '"0.3.99"'))
    result = _run(repo_copy)
    assert result.returncode != 0


def test_cross_pin_drift_detected(repo_copy: Path):
    """Server's flopscope==X.Y.Z pin must match root version."""
    pp = repo_copy / "flopscope-server" / "pyproject.toml"
    pp.write_text(pp.read_text().replace('"flopscope==0.3.0"', '"flopscope==0.3.99"'))
    result = _run(repo_copy)
    assert result.returncode != 0
    combined = (result.stdout + result.stderr).lower()
    assert "flopscope==" in combined or "cross-pin" in combined.replace(" ", "")


def test_client_pyproject_drift_detected(repo_copy: Path):
    """Client's pyproject version drift is caught."""
    pp = repo_copy / "flopscope-client" / "pyproject.toml"
    pp.write_text(pp.read_text().replace('version = "0.3.0"', 'version = "0.4.0"', 1))
    result = _run(repo_copy)
    assert result.returncode != 0
