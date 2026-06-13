"""ops.json must stay in sync with the registry (the generated single source)."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_ops_json_in_sync_with_registry():
    result = subprocess.run(
        [sys.executable, "scripts/generate_api_docs.py", "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "website/public/ops.json is out of sync with the registry. "
        "Run: uv run python scripts/generate_api_docs.py\n"
        + result.stdout
        + result.stderr
    )
