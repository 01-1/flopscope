"""Regression: ufunc wrappers must expose rich signatures, not (*args, **kwargs).

A latent bug overwrote ufunc wrapper __signature__ with numpy's opaque ufunc
signature; this guards against recurrence (it silently degraded the API docs).
"""
import inspect

import flopscope.numpy as fnp


def test_ufunc_wrapper_keeps_rich_signature():
    params = list(inspect.signature(fnp.add).parameters)
    assert params[:2] == ["x", "y"], f"ufunc wrapper signature degraded: {params}"


def test_non_ufunc_signature_unaffected():
    # sanity: a non-ufunc still has a real signature
    assert list(inspect.signature(fnp.broadcast_shapes).parameters)  # non-empty
