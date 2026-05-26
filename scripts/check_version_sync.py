#!/usr/bin/env python
"""Verify all flopscope package versions are in lockstep.

Run from the repo root:
    uv run python scripts/check_version_sync.py

Exits 0 if all seven version locations agree, 1 otherwise:
  - pyproject.toml [project].version            (root)
  - src/flopscope/__init__.py __version__       (leading X.Y.Z only)
  - flopscope-server/pyproject.toml version
  - flopscope-server/src/flopscope_server/__init__.py __version__
  - flopscope-server/pyproject.toml dependencies -> "flopscope==X.Y.Z"
  - flopscope-client/pyproject.toml version
  - flopscope-client/src/flopscope/__init__.py __version__
"""

from __future__ import annotations

import re
import sys
import tomllib
from collections import Counter
from pathlib import Path

_VERSION_RE = re.compile(r'__version__\s*=\s*f?"([0-9]+\.[0-9]+\.[0-9]+)')
_CROSS_PIN_RE = re.compile(r'"flopscope==([0-9]+\.[0-9]+\.[0-9]+)"')


def _read_pyproject_version(path: Path) -> str:
    with path.open("rb") as f:
        data = tomllib.load(f)
    return data["project"]["version"]


def _read_init_version(path: Path) -> str:
    """Extract leading X.Y.Z from an __version__ assignment, ignoring +suffix."""
    text = path.read_text()
    m = _VERSION_RE.search(text)
    if m is None:
        raise ValueError(f"no __version__ literal found in {path}")
    return m.group(1)


def _read_server_cross_pin(server_pyproject: Path) -> str:
    """Extract X.Y.Z from server's `flopscope==X.Y.Z` dependency."""
    text = server_pyproject.read_text()
    m = _CROSS_PIN_RE.search(text)
    if m is None:
        raise ValueError(
            f"no 'flopscope==X.Y.Z' pin found in {server_pyproject}; expected "
            "an exact pin in [project].dependencies"
        )
    return m.group(1)


def collect_versions(root: Path) -> dict[str, str]:
    """Return a dict of {location label: version string} for all 7 spots."""
    return {
        "pyproject.toml [project].version": _read_pyproject_version(
            root / "pyproject.toml"
        ),
        "src/flopscope/__init__.py __version__": _read_init_version(
            root / "src" / "flopscope" / "__init__.py"
        ),
        "flopscope-server/pyproject.toml [project].version": _read_pyproject_version(
            root / "flopscope-server" / "pyproject.toml"
        ),
        "flopscope-server/src/flopscope_server/__init__.py __version__": _read_init_version(
            root
            / "flopscope-server"
            / "src"
            / "flopscope_server"
            / "__init__.py"
        ),
        "flopscope-server/pyproject.toml flopscope== cross-pin": _read_server_cross_pin(
            root / "flopscope-server" / "pyproject.toml"
        ),
        "flopscope-client/pyproject.toml [project].version": _read_pyproject_version(
            root / "flopscope-client" / "pyproject.toml"
        ),
        "flopscope-client/src/flopscope/__init__.py __version__": _read_init_version(
            root / "flopscope-client" / "src" / "flopscope" / "__init__.py"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    root = Path.cwd()
    versions = collect_versions(root)
    distinct = set(versions.values())
    if len(distinct) == 1:
        v = next(iter(distinct))
        print(f"OK: all 7 version locations at {v}")
        return 0

    print("DRIFT DETECTED: package versions are not in lockstep.", file=sys.stderr)
    print("", file=sys.stderr)
    counts = Counter(versions.values())
    reference, _ = counts.most_common(1)[0]
    for label, ver in versions.items():
        marker = "  " if ver == reference else "!!"
        print(f"  {marker} {ver:12s}  {label}", file=sys.stderr)
    print("", file=sys.stderr)
    print(
        "Fix by editing the drifted file(s) above to match the reference version.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
