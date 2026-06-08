"""The five callback ops raise RemoteCallbackError (not an opaque msgpack
TypeError) when handed a Python callable on the client/server backend.

These tests need no live server: the error fires at encode time, before any
connection is made.
"""

import pytest
from flopscope._protocol import encode_request
from flopscope._registry_data import LOCAL_CALLBACK_OPS

import flopscope
from flopscope.errors import RemoteCallbackError


def test_local_callback_ops_set():
    assert LOCAL_CALLBACK_OPS == frozenset(
        {"apply_along_axis", "apply_over_axes", "piecewise", "fromfunction", "fromiter"}
    )


def test_encode_request_rejects_callables():
    # Precondition: msgpack genuinely cannot serialize a Python function.
    with pytest.raises((TypeError, ValueError)):
        encode_request("apply_along_axis", args=[lambda x: x], kwargs={})


@pytest.mark.parametrize("op", sorted(LOCAL_CALLBACK_OPS))
def test_callback_op_raises_remote_callback_error(op):
    fn = getattr(flopscope, op)
    with pytest.raises(RemoteCallbackError) as ei:
        fn(lambda *a, **k: 0.0)  # a Python callable can't cross the wire
    assert op in str(ei.value)
