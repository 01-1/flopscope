"""dot/inner N-D outer-product contractions must equal the equivalent einsum."""

from __future__ import annotations

import numpy as np

import flopscope.numpy as fnp
from flopscope import BudgetContext

rng = np.random.default_rng(0)


def _cost(fn, *a):
    with BudgetContext(flop_budget=int(1e20)) as bc:
        fn(*a)
    return bc.flops_used


def arr(*s):
    return fnp.asarray(rng.standard_normal(s).astype(np.float32))


def test_inner_nd_matches_value_and_is_exact():
    a, b = arr(3, 5, 4), arr(7, 4)  # contract last axis (4)
    out = np.asarray(fnp.inner(a, b))
    np.testing.assert_allclose(out, np.inner(np.asarray(a), np.asarray(b)), rtol=1e-4)
    assert _cost(fnp.inner, a, b) == 3 * 5 * 7 * (2 * 4 - 1)


def test_dot_nd_matches_value_and_is_exact():
    a, b = arr(3, 4), arr(5, 4, 6)  # np.dot contracts a[-1] with b[-2]
    out = np.asarray(fnp.dot(a, b))
    np.testing.assert_allclose(out, np.dot(np.asarray(a), np.asarray(b)), rtol=1e-4)
    assert _cost(fnp.dot, a, b) == 3 * 5 * 6 * (2 * 4 - 1)
