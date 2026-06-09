"""Client raises a clear RemoteSerializationError for non-serializable args."""

from __future__ import annotations

import pytest

import flopscope as fnp
from flopscope.errors import RemoteCallbackError, RemoteSerializationError


def test_generator_arg_raises_remote_serialization_error():
    with pytest.raises(RemoteSerializationError) as excinfo:
        fnp.concatenate([(x for x in range(3))])
    assert "generator" in str(excinfo.value)
    assert "concatenate" in str(excinfo.value)


def test_callback_op_still_raises_remote_callback_error():
    with pytest.raises(RemoteCallbackError):
        fnp.apply_along_axis(lambda r: r.sum(), 0, [[1, 2], [3, 4]])
