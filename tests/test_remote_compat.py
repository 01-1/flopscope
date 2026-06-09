"""Tests for in-process ↔ remote (client/server) divergence feedback."""

from __future__ import annotations

import flopscope as flops

_CALLBACK_OPS = frozenset(
    {"apply_along_axis", "apply_over_axes", "fromfunction", "fromiter", "piecewise"}
)


def test_remote_unsupported_ops_returns_frozenset_of_callback_ops():
    ops = flops.remote_unsupported_ops()
    assert isinstance(ops, frozenset)
    assert ops == _CALLBACK_OPS


def test_remote_unsupported_ops_matches_registry_flag():
    from flopscope._registry import REGISTRY

    assert flops.remote_unsupported_ops() == frozenset(
        name for name, entry in REGISTRY.items() if entry.get("local_callback")
    )
