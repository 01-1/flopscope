"""Budget-integrity regression: participant-reachable control cannot zero the bill."""

from __future__ import annotations

import struct

import msgpack
import pytest

from flopscope_server._server import FlopscopeServer

pytestmark = pytest.mark.security

TOKEN = "test-control-token"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_budget_context():
    """Ensure any lingering BudgetContext from a previous test is exited.

    Tests that open a session but don't close it leave an active BudgetContext
    in process state. This fixture tears that down after each test so subsequent
    tests can open their own sessions without hitting "Cannot nest BudgetContexts".
    """
    yield
    from flopscope._budget import get_active_budget
    ctx = get_active_budget()
    if ctx is not None:
        try:
            ctx.__exit__(None, None, None)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resp(raw):
    """Unpack a msgpack response to a plain dict (string keys)."""
    return msgpack.unpackb(raw, raw=False, strict_map_key=False)


def _server_no_token():
    """Server with NO control_token kwarg (tests the pre-fix baseline)."""
    return FlopscopeServer(url="inproc://test-integrity-no-token")


def _server_with_token():
    """Server configured with a control token (production posture).

    The server requires the control token for budget_open/budget_close; a
    request without it is rejected with UnauthorizedControlError and mutates
    no state.
    """
    return FlopscopeServer(url="inproc://test-integrity-token", control_token=TOKEN)


# ---------------------------------------------------------------------------
# Op helpers
# ---------------------------------------------------------------------------


def _open_budget(srv, **extra):
    """Send budget_open with optional extra kwargs (e.g. control_token)."""
    msg = {"op": "budget_open", "flop_budget": 10**9}
    msg.update(extra)
    return _resp(srv._process_request(msgpack.packb(msg, use_bin_type=True)))


def _add_cost(srv):
    """Run one counted op: elementwise add of two 1000-element float32 arrays.

    Expected cost: 1000 FLOPs (one add per element).

    Uses create_from_data so the arrays are deterministic and no prior op is needed.
    The handle key in the metadata response is "id" (per ArrayStore.metadata).
    """
    data = struct.pack("<1000f", *([1.0] * 1000))

    # Store first array
    r1 = _resp(
        srv._process_request(
            msgpack.packb(
                {"op": "create_from_data", "args": [data, [1000], "float32"]},
                use_bin_type=True,
            )
        )
    )
    assert r1["status"] == "ok", f"create_from_data failed: {r1}"
    # ArrayStore.metadata() returns {"id": handle, "shape": ..., "dtype": ...}
    handle1 = r1["result"]["id"]

    # Store second array
    r2 = _resp(
        srv._process_request(
            msgpack.packb(
                {"op": "create_from_data", "args": [data, [1000], "float32"]},
                use_bin_type=True,
            )
        )
    )
    assert r2["status"] == "ok", f"create_from_data failed: {r2}"
    handle2 = r2["result"]["id"]

    # Elementwise add — bills 1000 FLOPs
    add_resp = _resp(
        srv._process_request(
            msgpack.packb(
                {
                    "op": "add",
                    "args": [{"__handle__": handle1}, {"__handle__": handle2}],
                },
                use_bin_type=True,
            )
        )
    )
    assert add_resp["status"] == "ok", f"add op failed: {add_resp}"


def _flops_used(srv):
    """Query flops_used from budget_status."""
    r = _resp(srv._process_request(msgpack.packb({"op": "budget_status"}, use_bin_type=True)))
    assert r["status"] == "ok", f"budget_status failed: {r}"
    return r["result"]["flops_used"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_multiplier_zero_is_ignored():
    """The server must ignore any client-supplied flop_multiplier; cost is flop_cost × per-op weight only.

    A client sending flop_multiplier=0.0 must still be billed the real FLOPs.

    This test constructs the server with NO control_token so it does not hit
    the TypeError from the (not-yet-added) control_token kwarg.
    """
    srv = _server_no_token()
    # Open with flop_multiplier=0.0 — the exploit lever
    r = _open_budget(srv, flop_multiplier=0.0)
    assert r["status"] == "ok", f"budget_open failed: {r}"

    _add_cost(srv)

    used = _flops_used(srv)
    assert used >= 1000, (
        f"multiplier=0.0 must NOT zero the bill; got flops_used={used}. "
        "The server should ignore the client-supplied multiplier."
    )


def test_control_requires_token():
    """budget_open without the control token must be rejected with UnauthorizedControlError.

    The server requires the control token for budget_open; a request without
    it is rejected with UnauthorizedControlError and no session is created.
    """
    srv = _server_with_token()

    # Send budget_open WITHOUT a control_token — must be rejected
    bad = _resp(
        srv._process_request(
            msgpack.packb(
                {"op": "budget_open", "flop_budget": 10**9},
                use_bin_type=True,
            )
        )
    )
    assert bad["status"] == "error", f"expected error but got: {bad}"
    assert bad["error_type"] == "UnauthorizedControlError", (
        f"expected UnauthorizedControlError, got {bad.get('error_type')!r}"
    )

    # No session should have been created — budget_status must also error
    status = _resp(
        srv._process_request(msgpack.packb({"op": "budget_status"}, use_bin_type=True))
    )
    assert status["status"] == "error", (
        f"expected error from budget_status (no session), got: {status}"
    )


def test_close_requires_token():
    """budget_close without the token must be rejected; session must remain open.

    The server requires the control token for budget_close; a request without
    it is rejected with UnauthorizedControlError and the session remains open,
    continuing to accumulate costs normally.
    """
    srv = _server_with_token()

    # Open a legitimate session WITH the token
    r = _open_budget(srv, control_token=TOKEN)
    assert r["status"] == "ok", f"budget_open with token failed: {r}"

    # Attempt to close WITHOUT the token — must be rejected
    bad = _resp(
        srv._process_request(msgpack.packb({"op": "budget_close"}, use_bin_type=True))
    )
    assert bad["status"] == "error", f"expected error but got: {bad}"
    assert bad["error_type"] == "UnauthorizedControlError", (
        f"expected UnauthorizedControlError, got {bad.get('error_type')!r}"
    )

    # Session must still be open — cost should accumulate normally
    _add_cost(srv)
    used = _flops_used(srv)
    assert used >= 1000, (
        f"session should be intact after rejected close; got flops_used={used}"
    )


def test_token_fd_delivery():
    """--token-fd writes a token the parent can read; control then requires it."""
    import os, subprocess, sys, time, msgpack, zmq, secrets as _sec

    r, w = os.pipe()
    # IPC Unix-socket paths must be <104 chars; use /tmp with a short unique name.
    sock_path = f"/tmp/fts-{_sec.token_hex(6)}.sock"
    sock_url = f"ipc://{sock_path}"
    proc = subprocess.Popen(
        [sys.executable, "-m", "flopscope_server", "--url", sock_url,
         "--timeout", "30", "--token-fd", str(w)],
        pass_fds=(w,),
    )
    os.close(w)
    token = os.read(r, 4096).decode().strip()
    os.close(r)
    try:
        assert len(token) == 64  # token_hex(32)
        for _ in range(100):
            if os.path.exists(sock_path):
                break
            time.sleep(0.05)
        ctx = zmq.Context.instance()
        s = ctx.socket(zmq.REQ); s.setsockopt(zmq.RCVTIMEO, 5000); s.connect(sock_url)
        import flopscope
        s.send(msgpack.packb({"op": "hello", "kwargs":
            {"client_version": flopscope.__version__.split("+", 1)[0]}}, use_bin_type=True))
        s.recv()
        # open WITHOUT token → rejected
        s.send(msgpack.packb({"op": "budget_open", "kwargs": {"flop_budget": 10**9}},
                             use_bin_type=True))
        assert msgpack.unpackb(s.recv(), raw=False)["error_type"] == "UnauthorizedControlError"
        # open WITH token → ok
        s.send(msgpack.packb({"op": "budget_open",
            "kwargs": {"flop_budget": 10**9, "control_token": token}}, use_bin_type=True))
        assert msgpack.unpackb(s.recv(), raw=False)["status"] == "ok"
        s.close()
    finally:
        proc.terminate(); proc.wait(timeout=5)
        try:
            os.unlink(sock_path)
        except FileNotFoundError:
            pass
