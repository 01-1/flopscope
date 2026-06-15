"""Tests for server-side RandomState and SeedSequence handle pack/resolve."""

from __future__ import annotations

import numpy as np
import pytest
from flopscope_server._request_handler import RequestHandler
from flopscope_server._session import Session

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def session():
    s = Session(flop_budget=1_000_000)
    yield s
    if s.is_open:
        s.close()


@pytest.fixture()
def handler(session):
    return RequestHandler(session)


# ---------------------------------------------------------------------------
# RandomState pack + resolve round-trip
# ---------------------------------------------------------------------------


def test_pack_randomstate_returns_rs_id(handler):
    rs = np.random.RandomState(0)
    packed = handler._pack_result(rs)
    assert packed["status"] == "ok"
    assert "rs_id" in packed["result"]


def test_resolve_randomstate_round_trip(handler):
    rs = np.random.RandomState(0)
    packed = handler._pack_result(rs)
    rs_id = packed["result"]["rs_id"]
    resolved = handler._resolve_arg({"__rs__": rs_id})
    assert isinstance(resolved, np.random.RandomState)


def test_pack_randomstate_bytes_key_resolve(handler):
    """_resolve_arg must handle bytes keys from msgpack."""
    rs = np.random.RandomState(42)
    packed = handler._pack_result(rs)
    rs_id = packed["result"]["rs_id"]
    resolved = handler._resolve_arg({b"__rs__": rs_id.encode("utf-8")})
    assert isinstance(resolved, np.random.RandomState)


# ---------------------------------------------------------------------------
# SeedSequence pack + resolve round-trip
# ---------------------------------------------------------------------------


def test_pack_seedsequence_returns_seq_id(handler):
    seq = np.random.SeedSequence(123)
    packed = handler._pack_result(seq)
    assert packed["status"] == "ok"
    assert "seq_id" in packed["result"]


def test_resolve_seedsequence_round_trip(handler):
    seq = np.random.SeedSequence(123)
    packed = handler._pack_result(seq)
    seq_id = packed["result"]["seq_id"]
    resolved = handler._resolve_arg({"__seq__": seq_id})
    assert isinstance(resolved, np.random.SeedSequence)


def test_pack_seedsequence_bytes_key_resolve(handler):
    """_resolve_arg must handle bytes keys from msgpack."""
    seq = np.random.SeedSequence(456)
    packed = handler._pack_result(seq)
    seq_id = packed["result"]["seq_id"]
    resolved = handler._resolve_arg({b"__seq__": seq_id.encode("utf-8")})
    assert isinstance(resolved, np.random.SeedSequence)


# ---------------------------------------------------------------------------
# Existing Generator handling must be unaffected
# ---------------------------------------------------------------------------


def test_generator_pack_unaffected(handler):
    """Existing Generator branch still returns gen_id (not rs_id or seq_id)."""
    rng = np.random.default_rng(0)
    packed = handler._pack_result(rng)
    assert packed["status"] == "ok"
    assert "gen_id" in packed["result"]
    assert "rs_id" not in packed["result"]
    assert "seq_id" not in packed["result"]


def test_generator_resolve_unaffected(handler):
    """Existing __gen__ resolve still works after adding rs/seq branches."""
    rng = np.random.default_rng(0)
    packed = handler._pack_result(rng)
    gen_id = packed["result"]["gen_id"]
    resolved = handler._resolve_arg({"__gen__": gen_id})
    assert isinstance(resolved, np.random.Generator)


# ---------------------------------------------------------------------------
# RandomState method dispatch
# ---------------------------------------------------------------------------


def test_randomstate_method_dispatch_returns_array(handler):
    import flopscope.numpy as fnp

    rs = fnp.random.RandomState(0)  # _CountedRandomState
    handle = handler._session.store_generator(rs)
    budget_before = handler._session.budget_remaining
    resp = handler.handle(
        {
            "op": "RandomState.normal",
            "args": [{"__rs__": handle}],
            "kwargs": {"size": 8},
        }
    )
    assert resp["status"] == "ok"
    assert "id" in resp["result"]  # array handle key from array_metadata()
    # _CountedRandomState must bill FLOPs for the sample; budget must decrease.
    assert handler._session.budget_remaining < budget_before


def test_randomstate_method_rejects_unknown(handler):
    rs_handle = handler._session.store_generator(np.random.RandomState(0))
    resp = handler.handle(
        {
            "op": "RandomState.not_a_method",
            "args": [{"__rs__": rs_handle}],
            "kwargs": {},
        }
    )
    assert resp["status"] == "error"


def test_seedsequence_generate_state_dispatch(handler):
    seq_handle = handler._session.store_generator(np.random.SeedSequence(123))
    resp = handler.handle(
        {
            "op": "SeedSequence.generate_state",
            "args": [{"__seq__": seq_handle}, 4],
            "kwargs": {},
        }
    )
    assert resp["status"] == "ok"
    assert "id" in resp["result"]  # array handle key from array_metadata()


def test_randomstate_shuffle_is_rejected(handler):
    """shuffle must be excluded from the allowlist (in-place array mutator)."""
    # Store an array for the handle argument
    arr = np.arange(10, dtype=np.float64)
    arr_handle = handler._session.store_array(arr)
    rs_handle = handler._session.store_generator(np.random.RandomState(0))
    resp = handler.handle(
        {
            "op": "RandomState.shuffle",
            "args": [{"__rs__": rs_handle}, {"__handle__": arr_handle}],
            "kwargs": {},
        }
    )
    assert resp["status"] == "error"
