"""Unit tests (server-free) for client/server boundary timing attribution.

Billing rule: residual = wall − dispatch (only residual is billed). A round-trip
that grows `total_dispatch_ns` is absorbed into overhead and NOT billed. These
tests inject fakes so they need no server (and no numpy — the client is numpy-free).
"""

import struct
import time

import flopscope._connection as conn_mod
import flopscope._dispatch as dispatch

import flopscope as fnp
from flopscope import _codec

_DELAY = 0.03  # 30 ms simulated transfer/compute per round-trip


class _FakeConn:
    """Replaces the connection singleton; every round-trip costs _DELAY and
    returns a valid stored-array-handle response."""

    def send_recv(self, raw):  # noqa: ARG002 - request body irrelevant to the fake
        time.sleep(_DELAY)
        return {"result": {"id": "a0", "shape": [8, 8], "dtype": "float64"}}


def test_load_ingress_is_dispatch_not_residual(tmp_path, monkeypatch):
    """flops.load must bill its parse + transfer to dispatch/overhead, not the
    participant's residual. Regression for the _io send_recv-outside-span leak."""
    monkeypatch.setattr(conn_mod, "_connection", _FakeConn())

    flat = [float(i) for i in range(64)]
    npy = tmp_path / "w.npy"
    npy.write_bytes(_codec.write_npy("float64", (8, 8), struct.pack("<64d", *flat)))

    dispatch.reset_dispatch()
    before = dispatch.total_dispatch_ns()
    fnp.load(str(npy))
    grew = dispatch.total_dispatch_ns() - before

    assert grew >= _DELAY * 1e9 * 0.5, (
        "load() ingress not attributed to dispatch — it leaks into billed residual"
    )


def test_array_ingress_is_dispatch_not_residual(monkeypatch):
    """Control: fnp.array(list) already attributes its ingress to dispatch."""
    monkeypatch.setattr(conn_mod, "_connection", _FakeConn())

    dispatch.reset_dispatch()
    before = dispatch.total_dispatch_ns()
    fnp.array([[float(i) for i in range(8)] for _ in range(8)])
    grew = dispatch.total_dispatch_ns() - before

    assert grew >= _DELAY * 1e9 * 0.5


def test_send_recv_self_times_transport_as_dispatch(monkeypatch):
    """Connection.send_recv must attribute its own transport to dispatch, so even
    an UNDECORATED caller never bills the round-trip to residual (structural floor)."""
    import msgpack
    from flopscope._connection import Connection

    ok = msgpack.packb(
        {"status": "ok", "result": {"id": "a0", "shape": [2, 2], "dtype": "float64"}},
        use_bin_type=True,
    )

    class _FakeSock:
        def send(self, data):  # noqa: ARG002
            pass

        def recv(self):
            time.sleep(_DELAY)
            return ok

    conn = Connection()
    monkeypatch.setattr(conn, "_ensure_connected", lambda: _FakeSock())
    monkeypatch.setattr(conn, "_ensure_handshaked", lambda: None)

    dispatch.reset_dispatch()
    before = dispatch.total_dispatch_ns()
    resp = conn.send_recv(b"ignored-request")
    grew = dispatch.total_dispatch_ns() - before

    assert resp["result"]["id"] == "a0"
    assert grew >= _DELAY * 1e9 * 0.5, (
        "send_recv transport not self-timed into dispatch"
    )
