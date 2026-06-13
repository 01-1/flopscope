"""ops.json cost-model fields must stay in sync with the registry.

`generate_api_docs.py --check` compares every field except ``summary`` (which is
sourced from the installed numpy's docstrings and legitimately varies across the
numpy-version matrix). This test therefore also doubles as a guarantee that the
cost model itself is numpy-version-independent: it runs in every matrixed test job
(numpy 2.0-2.4) and must pass on all of them.
"""

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
