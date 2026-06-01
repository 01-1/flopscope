"""Integration tests for fnp.random.default_rng() over the client/server boundary.

Regression coverage for the grader failure where
``fnp.random.default_rng(seed)`` raised
``FlopscopeServerError: failed to serialize response: TypeError`` because the
server could not serialize a numpy Generator (no remote-handle support).

Runs a real FlopscopeServer in a subprocess (its own port so this file is
isolation-safe) and drives the full request/response chain.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

import pytest

_WORKTREE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CLIENT_SRC = os.path.join(_WORKTREE, "flopscope-client", "src")
_SERVER_SRC = os.path.join(_WORKTREE, "flopscope-server", "src")
_REAL_SRC = os.path.join(_WORKTREE, "src")
_VENV_PYTHON = os.path.join(_WORKTREE, ".venv", "bin", "python")

for _p in (_CLIENT_SRC,):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_SERVER_URL = "tcp://127.0.0.1:15557"

_SERVER_SCRIPT = f"""
import sys
sys.path.insert(0, {_REAL_SRC!r})
sys.path.insert(0, {_SERVER_SRC!r})
from flopscope_server._server import FlopscopeServer
server = FlopscopeServer(url={_SERVER_URL!r})
print("SERVER_READY", flush=True)
server.run()
"""


@pytest.fixture(scope="session", autouse=True)
def _start_server():
    os.environ["FLOPSCOPE_SERVER_URL"] = _SERVER_URL
    proc = subprocess.Popen(
        [_VENV_PYTHON, "-c", _SERVER_SCRIPT],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    line = proc.stdout.readline()
    assert "SERVER_READY" in line, f"Server failed to start: {line}"
    time.sleep(0.3)
    yield proc
    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=5)


@pytest.fixture(autouse=True)
def _reset_client():
    from flopscope._connection import reset_connection

    reset_connection()
    yield
    reset_connection()


def test_default_rng_uniform_returns_array():
    """default_rng(seed).uniform(...) must work across the boundary (RED until fix)."""
    import flopscope as we

    with we.BudgetContext(flop_budget=10_000_000):
        rng = we.random.default_rng(0)
        out = rng.uniform(0.0, 1.0, size=(2, 3))
        assert out.shape == (2, 3)
        vals = out.tolist()
        assert all(0.0 <= v <= 1.0 for row in vals for v in row)


def test_default_rng_reproducible():
    """Same seed must yield an identical stream (server-side determinism).

    The bit-for-bit match against numpy's own default_rng is asserted in the
    server-side suite, where numpy is importable (the client is numpy-free).
    """
    import flopscope as we

    with we.BudgetContext(flop_budget=10_000_000):
        a = we.random.default_rng(7).standard_normal(size=(5,)).tolist()
        b = we.random.default_rng(7).standard_normal(size=(5,)).tolist()
    assert a == b
    assert any(x != 0.0 for x in a)
