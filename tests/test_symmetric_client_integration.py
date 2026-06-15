"""Integration test: symmetric ops round-trip through the client→server boundary.

Uses the same in-process harness as ``tests/test_serialization_parity.py`` and
``tests/test_symmetry_transport.py``: a real ``Session`` + ``RequestHandler``
from ``flopscope-server`` replaces the network, and the client-side
``_encode_arg`` / ``_result_from_response`` helpers from ``flopscope-client``
handle serialisation.  This exercises the FULL client→server dispatch path
(arg encoding, SymmetryGroup wire format, result packing, symmetry metadata
round-trip) without requiring ZMQ or msgpack to be installed.

Covered ops
-----------
* ``symmetrize``   — Reynolds projection, result carries `.symmetry`
* ``as_symmetric`` — tag-only, result carries `.symmetry`
* ``is_symmetric`` — predicate, returns boolean scalar (RemoteScalar)
* ``random.symmetric`` — sample + project, result carries `.symmetry`
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path setup — mirrors test_serialization_parity.py exactly
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent
_SERVER_SRC = str(_ROOT / "flopscope-server" / "src")
_CLIENT_SRC = str(_ROOT / "flopscope-client" / "src")

if _SERVER_SRC not in sys.path:
    sys.path.insert(0, _SERVER_SRC)

from flopscope_server._request_handler import (  # pyright: ignore[reportMissingImports]
    RequestHandler,  # noqa: E402
)
from flopscope_server._session import (  # pyright: ignore[reportMissingImports]
    Session,  # noqa: E402
)


def _load_client_module(rel_path: str, module_name: str):
    if module_name in sys.modules:
        return sys.modules[module_name]
    module_file = Path(_CLIENT_SRC) / rel_path
    spec = importlib.util.spec_from_file_location(module_name, module_file)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_load_client_module("flopscope/_constants.py", "flopscope._constants")
_load_client_module("flopscope/_math_compat.py", "flopscope._math_compat")
_client_perm_group = _load_client_module(
    "flopscope/_perm_group.py", "flopscope._perm_group"
)
_load_client_module("flopscope/_dispatch.py", "flopscope._dispatch")
_client_remote_array = _load_client_module(
    "flopscope/_remote_array.py", "flopscope._remote_array"
)

RemoteArray = _client_remote_array.RemoteArray
RemoteScalar = _client_remote_array.RemoteScalar
_encode_arg = _client_remote_array._encode_arg
_result_from_response = _client_remote_array._result_from_response
ClientSymmetryGroup = _client_perm_group.SymmetryGroup


# ---------------------------------------------------------------------------
# Fixtures — identical pattern to test_serialization_parity.py
# ---------------------------------------------------------------------------


@pytest.fixture()
def handler_session():
    session = Session(flop_budget=10**12)
    handler = RequestHandler(session)
    yield session, handler
    if session.is_open:
        session.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dispatch(handler: RequestHandler, session: Session, op: str, *args, **kwargs):
    """Encode *args*/*kwargs* with the client encoder, dispatch to the handler,
    and decode the response with the client decoder.

    Arrays stored in the session are passed through as handle-dicts so the
    server resolves them from its ArrayStore — exactly what the real client does.
    """
    encoded_args = []
    for a in args:
        if isinstance(a, np.ndarray):
            handle = session.store_array(a)
            encoded_args.append({"__handle__": handle})
        else:
            encoded_args.append(_encode_arg(a))
    encoded_kwargs = {}
    for k, v in kwargs.items():
        if isinstance(v, np.ndarray):
            handle = session.store_array(v)
            encoded_kwargs[k] = {"__handle__": handle}
        else:
            encoded_kwargs[k] = _encode_arg(v)

    request = {"op": op, "args": encoded_args, "kwargs": encoded_kwargs}
    resp = handler.handle(request)
    assert resp.get("status") == "ok", (
        f"op={op!r} failed: {resp.get('error_type')}: {resp.get('message')}"
    )
    return _result_from_response(resp)


def _sym_payload():
    return {"axes": [0, 1], "generators": [[1, 0]]}


def _client_G():
    return ClientSymmetryGroup.from_payload(_sym_payload())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSymmetrizeRoundTrip:
    """symmetrize → result is a RemoteArray carrying .symmetry metadata."""

    def test_result_is_remote_array(self, handler_session):
        session, handler = handler_session
        G = ClientSymmetryGroup.symmetric(axes=(0, 1))
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = _dispatch(handler, session, "symmetrize", data, symmetry=G)
        assert isinstance(result, RemoteArray)

    def test_symmetry_metadata_round_trips(self, handler_session):
        session, handler = handler_session
        G = ClientSymmetryGroup.symmetric(axes=(0, 1))
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = _dispatch(handler, session, "symmetrize", data, symmetry=G)
        assert result.symmetry is not None
        assert result.symmetry == _client_G()

    def test_shape_preserved(self, handler_session):
        session, handler = handler_session
        G = ClientSymmetryGroup.symmetric(axes=(0, 1))
        data = np.ones((5, 5))
        result = _dispatch(handler, session, "symmetrize", data, symmetry=G)
        assert result.shape == (5, 5)

    def test_symmetrized_values_are_symmetric(self, handler_session):
        """The server-side Reynolds averaging must produce a genuinely symmetric
        matrix; verify by fetching the result back as raw data and checking
        the off-diagonal elements."""
        session, handler = handler_session
        G = ClientSymmetryGroup.symmetric(axes=(0, 1))
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = _dispatch(handler, session, "symmetrize", data, symmetry=G)
        # Fetch the array data back through the server
        fetch_resp = handler.handle({"op": "fetch", "id": result.handle_id})
        assert fetch_resp.get("status") == "ok"
        arr = np.frombuffer(fetch_resp["data"], dtype=fetch_resp["dtype"]).reshape(
            fetch_resp["shape"]
        )
        np.testing.assert_allclose(arr, arr.T, atol=1e-12)


class TestAsSymmetricRoundTrip:
    """as_symmetric → tag-only; result carries .symmetry without modifying data."""

    def test_result_is_remote_array(self, handler_session):
        session, handler = handler_session
        G = ClientSymmetryGroup.symmetric(axes=(0, 1))
        data = np.array([[1.0, 2.0], [2.0, 3.0]])  # already symmetric
        result = _dispatch(handler, session, "as_symmetric", data, symmetry=G)
        assert isinstance(result, RemoteArray)

    def test_symmetry_metadata_round_trips(self, handler_session):
        session, handler = handler_session
        G = ClientSymmetryGroup.symmetric(axes=(0, 1))
        data = np.array([[1.0, 2.0], [2.0, 3.0]])
        result = _dispatch(handler, session, "as_symmetric", data, symmetry=G)
        assert result.symmetry is not None
        assert result.symmetry == _client_G()

    def test_shape_preserved(self, handler_session):
        session, handler = handler_session
        G = ClientSymmetryGroup.symmetric(axes=(0, 1))
        data = np.eye(4)
        result = _dispatch(handler, session, "as_symmetric", data, symmetry=G)
        assert result.shape == (4, 4)


class TestIsSymmetricRoundTrip:
    """is_symmetric → returns a boolean scalar (RemoteScalar with dtype='bool')."""

    def test_true_for_symmetric_input(self, handler_session):
        session, handler = handler_session
        G = ClientSymmetryGroup.symmetric(axes=(0, 1))
        data = np.array([[1.0, 2.0], [2.0, 3.0]])
        result = _dispatch(handler, session, "is_symmetric", data, symmetry=G)
        assert isinstance(result, RemoteScalar)
        assert bool(result) is True

    def test_false_for_asymmetric_input(self, handler_session):
        session, handler = handler_session
        G = ClientSymmetryGroup.symmetric(axes=(0, 1))
        data = np.array([[1.0, 2.0], [3.0, 4.0]])  # not symmetric
        result = _dispatch(handler, session, "is_symmetric", data, symmetry=G)
        assert isinstance(result, RemoteScalar)
        assert bool(result) is False

    def test_result_dtype_is_bool(self, handler_session):
        session, handler = handler_session
        G = ClientSymmetryGroup.symmetric(axes=(0, 1))
        data = np.eye(3)
        result = _dispatch(handler, session, "is_symmetric", data, symmetry=G)
        assert result.dtype == "bool"


class TestRandomSymmetricRoundTrip:
    """random.symmetric → sampled + projected; result carries .symmetry."""

    def test_result_is_remote_array(self, handler_session):
        session, handler = handler_session
        G = ClientSymmetryGroup.symmetric(axes=(0, 1))
        result = _dispatch(handler, session, "random.symmetric", [4, 4], G)
        assert isinstance(result, RemoteArray)

    def test_symmetry_metadata_round_trips(self, handler_session):
        session, handler = handler_session
        G = ClientSymmetryGroup.symmetric(axes=(0, 1))
        result = _dispatch(handler, session, "random.symmetric", [4, 4], G)
        assert result.symmetry is not None
        assert result.symmetry == _client_G()

    def test_shape_is_correct(self, handler_session):
        session, handler = handler_session
        G = ClientSymmetryGroup.symmetric(axes=(0, 1))
        result = _dispatch(handler, session, "random.symmetric", [4, 4], G)
        assert result.shape == (4, 4)

    def test_output_is_actually_symmetric(self, handler_session):
        """Fetch the raw data back and confirm the server produced a symmetric
        matrix (Reynolds projection must have run)."""
        session, handler = handler_session
        G = ClientSymmetryGroup.symmetric(axes=(0, 1))
        result = _dispatch(handler, session, "random.symmetric", [4, 4], G)
        fetch_resp = handler.handle({"op": "fetch", "id": result.handle_id})
        assert fetch_resp.get("status") == "ok"
        arr = np.frombuffer(fetch_resp["data"], dtype=fetch_resp["dtype"]).reshape(
            fetch_resp["shape"]
        )
        np.testing.assert_allclose(arr, arr.T, atol=1e-12)


class TestSymmetricOpsEndToEnd:
    """Full pipeline: symmetrize → is_symmetric → as_symmetric, chained through
    the same Session so handles from one call are usable in the next."""

    def test_symmetrize_then_is_symmetric(self, handler_session):
        """symmetrize output handle is fed back into is_symmetric."""
        session, handler = handler_session
        G = ClientSymmetryGroup.symmetric(axes=(0, 1))
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        sym_result = _dispatch(handler, session, "symmetrize", data, symmetry=G)
        # Pass the RemoteArray handle back as input to is_symmetric
        is_sym = _dispatch(handler, session, "is_symmetric", sym_result, symmetry=G)
        assert bool(is_sym) is True

    def test_random_symmetric_then_is_symmetric(self, handler_session):
        """random.symmetric output is confirmed symmetric by is_symmetric."""
        session, handler = handler_session
        G = ClientSymmetryGroup.symmetric(axes=(0, 1))
        rand_sym = _dispatch(handler, session, "random.symmetric", [3, 3], G)
        is_sym = _dispatch(handler, session, "is_symmetric", rand_sym, symmetry=G)
        assert bool(is_sym) is True

    def test_symmetrize_then_as_symmetric(self, handler_session):
        """Chain symmetrize → as_symmetric: both results carry symmetry metadata."""
        session, handler = handler_session
        G = ClientSymmetryGroup.symmetric(axes=(0, 1))
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        sym = _dispatch(handler, session, "symmetrize", data, symmetry=G)
        tagged = _dispatch(handler, session, "as_symmetric", sym, symmetry=G)
        assert sym.symmetry is not None
        assert tagged.symmetry is not None
        assert sym.symmetry == tagged.symmetry

    def test_budget_is_deducted(self, handler_session):
        """Ops must actually deduct from the session budget."""
        session, handler = handler_session
        budget_before = session.budget_remaining
        G = ClientSymmetryGroup.symmetric(axes=(0, 1))
        data = np.ones((4, 4))
        _dispatch(handler, session, "symmetrize", data, symmetry=G)
        assert session.budget_remaining < budget_before
