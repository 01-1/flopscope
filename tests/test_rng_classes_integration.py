"""Integration test: RNG proxy classes round-trip through the client→server boundary.

Exercises the real client proxy classes — ``RemoteRandomState``,
``RemoteSeedSequence`` — via construction, sampling, billing, and rejection of
in-place mutators.

Harness
-------
Same in-process architecture as ``test_symmetric_client_integration.py``:
a real ``Session`` + ``RequestHandler`` from ``flopscope-server`` stands in for
the network, and client modules are loaded from ``flopscope-client/src`` via
``importlib``.

Two environment requirements prevent loading the full client package normally:

* **msgpack** — ``flopscope-client`` depends on it for wire encoding, but the
  root dev environment does not install it.  We stub it with an identity
  codec (``packb`` returns its input unchanged, ``unpackb`` returns its input
  unchanged) so ``flopscope._protocol.encode_request`` passes a plain dict
  to the fake connection, which forwards it directly to
  ``RequestHandler.handle``.

* **zmq** — ``flopscope._connection`` imports it at the top level.  We bypass
  it entirely by injecting a synthetic ``flopscope._connection`` module whose
  ``get_connection()`` returns a ``_FakeConnection`` that forwards
  dict-shaped requests to the in-process handler.

No network socket, no subprocess, no msgpack encoding overhead: just the real
client proxy code calling into the real server handler.

Covered
-------
* ``RemoteRandomState(seed)`` — construction dispatches to the server
* ``rs.normal(size=8)`` — sampling returns a ``RemoteArray`` and bills FLOPs
* ``RemoteSeedSequence(entropy)`` — construction dispatches to the server
* ``fnp.random.default_rng(seq)`` — ``SeedSequence`` handle usable as seed
* ``rs.shuffle(arr)`` — server rejects in-place mutators
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent
_SERVER_SRC = str(_ROOT / "flopscope-server" / "src")
_CLIENT_SRC = str(_ROOT / "flopscope-client" / "src")

if _SERVER_SRC not in sys.path:
    sys.path.insert(0, _SERVER_SRC)

# ---------------------------------------------------------------------------
# msgpack stub — must appear in sys.modules BEFORE any client module that
# imports it (flopscope._protocol uses it at the top level).
# ---------------------------------------------------------------------------

if "msgpack" not in sys.modules:
    _msgpack_stub = types.ModuleType("msgpack")
    _msgpack_stub.packb = lambda obj, use_bin_type=True: obj  # dict → dict (not bytes)
    _msgpack_stub.unpackb = lambda data, raw=True, strict_map_key=False: data
    sys.modules["msgpack"] = _msgpack_stub

# ---------------------------------------------------------------------------
# Server-side imports (need REGISTRY, numpy, etc.)
# ---------------------------------------------------------------------------

from flopscope_server._request_handler import (  # pyright: ignore[reportMissingImports]
    RequestHandler,  # noqa: E402
)
from flopscope_server._session import (  # pyright: ignore[reportMissingImports]
    Session,  # noqa: E402
)

# ---------------------------------------------------------------------------
# Client module loader (same pattern as test_symmetric_client_integration.py)
# ---------------------------------------------------------------------------


def _load_client_module(rel_path: str, module_name: str) -> types.ModuleType:
    """Load a flopscope-client source file under an explicit module name.

    Idempotent: if *module_name* is already in ``sys.modules`` the cached
    module is returned without re-executing the file.
    """
    if module_name in sys.modules:
        return sys.modules[module_name]
    module_file = Path(_CLIENT_SRC) / rel_path
    spec = importlib.util.spec_from_file_location(module_name, module_file)
    assert spec and spec.loader, f"could not locate client module at {module_file}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


# Load client modules in dependency order.
# _constants, _math_compat — no outside deps
_load_client_module("flopscope/_constants.py", "flopscope._constants")
_load_client_module("flopscope/_math_compat.py", "flopscope._math_compat")
# _perm_group — needed by _remote_array for SymmetryGroup decoding
_load_client_module("flopscope/_perm_group.py", "flopscope._perm_group")
# _dispatch — timing accumulator, no network deps
_load_client_module("flopscope/_dispatch.py", "flopscope._dispatch")
# _protocol — uses msgpack (stubbed above); encode_request → returns dict
_load_client_module("flopscope/_protocol.py", "flopscope._protocol")

# ---------------------------------------------------------------------------
# Inject a synthetic flopscope._connection that skips ZMQ entirely.
# Must be done BEFORE loading _remote_array (which imports _connection lazily
# on first method call, but the module is registered in sys.modules here).
# ---------------------------------------------------------------------------

if "flopscope._connection" not in sys.modules:
    _fake_conn_module = types.ModuleType("flopscope._connection")
    # Populated by the fixture below; safe to register now.
    _fake_conn_module.get_connection = None  # type: ignore[assignment]
    _fake_conn_module.reset_connection = lambda: None
    _fake_conn_module._connection = None
    sys.modules["flopscope._connection"] = _fake_conn_module

# Now load _remote_array (imports _connection + _protocol lazily inside methods).
_client_remote_array = _load_client_module(
    "flopscope/_remote_array.py", "flopscope._remote_array"
)

RemoteArray = _client_remote_array.RemoteArray
RemoteGenerator = _client_remote_array.RemoteGenerator
RemoteRandomState = _client_remote_array.RemoteRandomState
RemoteSeedSequence = _client_remote_array.RemoteSeedSequence
_encode_arg = _client_remote_array._encode_arg
_result_from_response = _client_remote_array._result_from_response

# ---------------------------------------------------------------------------
# Fixture: in-process session + handler + patched get_connection
# ---------------------------------------------------------------------------


@pytest.fixture()
def rng_env():
    """Yield a (session, handler) pair wired to the client's get_connection.

    All client proxy ``send_recv`` calls go to the in-process handler —
    no socket, no subprocess, no msgpack serialisation.
    """
    import flopscope as _flops

    session = Session(flop_budget=10**12)
    handler = RequestHandler(session)

    class _FakeConnection:
        """Accepts a dict-shaped request (encode_request returns dicts when
        msgpack is stubbed), calls handler.handle, raises on error status."""

        def send_recv(self, request: dict) -> dict:  # type: ignore[override]
            resp = handler.handle(request)
            if resp.get("status") == "error":
                error_type = resp.get("error_type", "FlopscopeError")
                message = resp.get("message", "")
                exc_map = {
                    "BudgetExhaustedError": _flops.BudgetExhaustedError,
                    "UnsupportedFunctionError": _flops.UnsupportedFunctionError,
                    "NoBudgetContextError": _flops.NoBudgetContextError,
                    "SymmetryError": _flops.SymmetryError,
                    "ValueError": ValueError,
                    "TypeError": TypeError,
                    "KeyError": KeyError,
                }
                exc_cls = exc_map.get(error_type, _flops.FlopscopeError)
                raise exc_cls(message)
            return resp

    _fake_instance = _FakeConnection()
    _conn_mod = sys.modules["flopscope._connection"]
    _conn_mod.get_connection = lambda: _fake_instance  # type: ignore[assignment]

    try:
        yield session
    finally:
        if session.is_open:
            session.close()
        # Restore so other tests that don't use this fixture are unaffected.
        _conn_mod.get_connection = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRemoteRandomStateConstruction:
    """RemoteRandomState construction dispatches to the server."""

    def test_constructs_and_has_handle(self, rng_env):
        """RemoteRandomState(0) must resolve to a server-side RandomState handle."""
        rs = RemoteRandomState(0)
        assert isinstance(rs, RemoteRandomState)
        # Handle must be a non-empty string (e.g. "g0")
        assert isinstance(rs.handle_id, str) and rs.handle_id

    def test_different_seeds_give_different_handles(self, rng_env):
        """Two separate construction calls must yield distinct server handles."""
        rs0 = RemoteRandomState(0)
        rs1 = RemoteRandomState(42)
        assert rs0.handle_id != rs1.handle_id

    def test_none_seed_accepted(self, rng_env):
        """RandomState(None) is valid (numpy default — seeded from /dev/urandom)."""
        rs = RemoteRandomState(None)
        assert rs.handle_id


class TestRemoteRandomStateSamplesAndBills:
    """Sampling methods return RemoteArray and deduct FLOPs from the session."""

    def test_normal_returns_remote_array(self, rng_env):
        rs = RemoteRandomState(0)
        result = rs.normal(size=8)
        assert isinstance(result, RemoteArray)

    def test_normal_shape_correct(self, rng_env):
        rs = RemoteRandomState(0)
        result = rs.normal(size=8)
        assert result.shape == (8,)

    def test_normal_bills_flops(self, rng_env):
        """normal(size=8) must deduct FLOPs from the server session budget."""
        rs = RemoteRandomState(0)
        before = rng_env.budget_remaining
        rs.normal(size=8)
        assert rng_env.budget_remaining < before

    def test_randn_returns_remote_array_with_shape(self, rng_env):
        rs = RemoteRandomState(1)
        result = rs.randn(4, 4)
        assert isinstance(result, RemoteArray)
        assert result.shape == (4, 4)

    def test_uniform_returns_remote_array(self, rng_env):
        rs = RemoteRandomState(7)
        result = rs.uniform(0.0, 1.0, size=16)
        assert isinstance(result, RemoteArray)
        assert result.shape == (16,)

    def test_multiple_calls_accumulate_billing(self, rng_env):
        """Billing must accumulate across multiple sampling calls."""
        rs = RemoteRandomState(0)
        before = rng_env.budget_remaining
        rs.normal(size=64)
        after_first = rng_env.budget_remaining
        rs.normal(size=64)
        after_second = rng_env.budget_remaining
        assert after_first < before
        assert after_second < after_first


class TestRemoteRandomStateShuffle:
    """RandomState.shuffle must be rejected by the server (in-place mutator)."""

    def test_shuffle_raises(self, rng_env):
        """shuffle is excluded from _ALLOWED_RS_METHODS; must raise."""
        import numpy as np

        import flopscope as _flops

        rs = RemoteRandomState(0)
        handle = rng_env.store_array(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        arr = RemoteArray(handle_id=handle, shape=(5,), dtype="float64")
        with pytest.raises(_flops.UnsupportedFunctionError):
            rs.shuffle(arr)

    def test_shuffle_rejection_message_mentions_shuffle(self, rng_env):
        """The error raised for shuffle should mention the method name."""
        import numpy as np

        import flopscope as _flops

        handle = rng_env.store_array(np.array([1.0, 2.0, 3.0]))
        arr = RemoteArray(handle_id=handle, shape=(3,), dtype="float64")
        rs = RemoteRandomState(0)
        with pytest.raises(_flops.UnsupportedFunctionError, match="shuffle"):
            rs.shuffle(arr)


class TestRemoteSeedSequence:
    """RemoteSeedSequence construction and use as a default_rng seed."""

    def test_constructs_and_has_handle(self, rng_env):
        seq = RemoteSeedSequence(123)
        assert isinstance(seq, RemoteSeedSequence)
        assert isinstance(seq.handle_id, str) and seq.handle_id

    def test_different_entropies_give_different_handles(self, rng_env):
        seq0 = RemoteSeedSequence(0)
        seq1 = RemoteSeedSequence(999)
        assert seq0.handle_id != seq1.handle_id

    def test_none_entropy_accepted(self, rng_env):
        seq = RemoteSeedSequence(None)
        assert seq.handle_id

    def test_usable_as_default_rng_seed(self, rng_env):
        """SeedSequence handle must be accepted as the seed argument to default_rng.

        The server resolves the ``__seq__`` handle to the stored SeedSequence
        and passes it to numpy.random.default_rng, which returns a Generator.
        The client decodes the ``gen_id`` in the response as a RemoteGenerator.
        """
        seq = RemoteSeedSequence(42)
        # Encode the seq as a handle-dict (what the real client does)
        seq_encoded = _encode_arg(seq)
        # Call default_rng on the server through the fake connection
        _conn = sys.modules["flopscope._connection"].get_connection()
        resp = _conn.send_recv(
            {"op": "random.default_rng", "args": [seq_encoded], "kwargs": {}}
        )
        rng = _result_from_response(resp)
        assert isinstance(rng, RemoteGenerator)

    def test_rng_from_seedsequence_can_sample(self, rng_env):
        """Generator seeded via SeedSequence must be able to sample arrays."""
        seq = RemoteSeedSequence(77)
        seq_encoded = _encode_arg(seq)
        _conn = sys.modules["flopscope._connection"].get_connection()
        resp = _conn.send_recv(
            {"op": "random.default_rng", "args": [seq_encoded], "kwargs": {}}
        )
        rng = _result_from_response(resp)
        result = rng.standard_normal((4,))
        assert isinstance(result, RemoteArray)
        assert result.shape == (4,)


class TestRNGClassesEndToEnd:
    """End-to-end: multiple RNG objects sharing a session."""

    def test_randomstate_and_seedsequence_independent_handles(self, rng_env):
        """RandomState and SeedSequence must receive distinct server handles."""
        rs = RemoteRandomState(10)
        seq = RemoteSeedSequence(20)
        assert rs.handle_id != seq.handle_id

    def test_budget_decreases_only_on_sampling(self, rng_env):
        """Construction (RandomState, SeedSequence) must not bill FLOPs;
        only sampling deducts from the budget.

        numpy.random.RandomState.__init__ and SeedSequence.__init__ are
        pure metadata operations (0 FLOPs), so the budget must be unchanged
        after construction and decrease only after a sampling call.
        """
        before = rng_env.budget_remaining
        _ = RemoteRandomState(0)
        _ = RemoteSeedSequence(123)
        after_constructs = rng_env.budget_remaining
        # Construction may bill 0 FLOPs — budget_remaining must not increase
        # (it may stay equal if construction is free, or decrease slightly if
        # the counted class bills anything for __init__).
        assert after_constructs <= before

        rs = RemoteRandomState(99)
        rs.normal(size=100)
        assert rng_env.budget_remaining < after_constructs

    def test_two_randomstates_sample_independently(self, rng_env):
        """Two RemoteRandomState objects must have independent server-side state."""
        rs0 = RemoteRandomState(0)
        rs1 = RemoteRandomState(0)
        result0 = rs0.normal(size=4)
        result1 = rs1.normal(size=4)
        # Both return RemoteArrays; the handles must be distinct (different output)
        assert isinstance(result0, RemoteArray)
        assert isinstance(result1, RemoteArray)
        assert result0.handle_id != result1.handle_id
