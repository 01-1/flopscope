"""Tests for the server-side version handshake (`hello` op).

The hello op is handled at the FlopscopeServer level (alongside
budget_open / budget_close) because it's session-lifecycle: the
handshake must succeed before any budget session can be opened.
"""

from __future__ import annotations

import msgpack

import flopscope
from flopscope_server import _protocol
from flopscope_server._server import FlopscopeServer


def _leading_xyz(v: str) -> str:
    """Strip any +np... suffix from flopscope.__version__."""
    return v.split("+", 1)[0]


def test_hello_in_protocol_ops():
    """The `hello` op must be on the permitted op whitelist."""
    assert "hello" in _protocol._PROTOCOL_OPS
    assert "hello" in _protocol.WHITELIST


def test_handle_hello_matching_version_returns_ok():
    """Matching client_version → ok response with server_version."""
    server = FlopscopeServer.__new__(FlopscopeServer)
    server._session = None
    server._handler = None
    server._last_activity = 0.0

    client_version = _leading_xyz(flopscope.__version__)
    response_bytes = server._handle_hello(
        {"op": "hello", "kwargs": {"client_version": client_version}}
    )
    decoded = msgpack.unpackb(response_bytes, raw=False)
    assert decoded["status"] == "ok"
    assert decoded["server_version"] == client_version


def test_handle_hello_mismatch_returns_version_mismatch_error():
    """Mismatched client_version → error response with VersionMismatch."""
    server = FlopscopeServer.__new__(FlopscopeServer)
    server._session = None
    server._handler = None
    server._last_activity = 0.0

    response_bytes = server._handle_hello(
        {"op": "hello", "kwargs": {"client_version": "0.0.99"}}
    )
    decoded = msgpack.unpackb(response_bytes, raw=False)
    assert decoded["status"] == "error"
    assert decoded["error_type"] == "VersionMismatch"
    server_xyz = _leading_xyz(flopscope.__version__)
    assert "0.0.99" in decoded["message"]
    assert server_xyz in decoded["message"]


def test_handle_hello_missing_client_version_is_mismatch():
    """A hello with no client_version kwarg is treated as a VersionMismatch."""
    server = FlopscopeServer.__new__(FlopscopeServer)
    server._session = None
    server._handler = None
    server._last_activity = 0.0

    response_bytes = server._handle_hello({"op": "hello", "kwargs": {}})
    decoded = msgpack.unpackb(response_bytes, raw=False)
    assert decoded["status"] == "error"
    assert decoded["error_type"] == "VersionMismatch"


def test_hello_works_before_budget_open():
    """The hello op must succeed even when no session is active."""
    server = FlopscopeServer.__new__(FlopscopeServer)
    server._session = None
    server._handler = None
    server._last_activity = 0.0

    client_version = _leading_xyz(flopscope.__version__)
    raw_request = msgpack.packb(
        {"op": "hello", "kwargs": {"client_version": client_version}},
        use_bin_type=True,
    )
    response_bytes = server._process_request(raw_request)
    decoded = msgpack.unpackb(response_bytes, raw=False)
    assert decoded["status"] == "ok"
    assert decoded["server_version"] == client_version
    # And no session was opened as a side effect.
    assert server._session is None
