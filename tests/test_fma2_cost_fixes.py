"""Charged-vs-honest FMA=2 cost tests for the audit fixes (one PR).

Charged = flops_used inside a BudgetContext (cost = flop_cost * weight),
under autoloaded packaged weights. Each test asserts the honest FMA=2 count.
"""
from __future__ import annotations

import numpy as np

import flopscope as f
import flopscope.numpy as fnp


def cost(fn, *args, **kwargs) -> int:
    with f.BudgetContext(flop_budget=10**18, quiet=True) as b:
        fn(*args, **kwargs)
        return b.flops_used


def test_tensordot_matches_einsum_partial_contraction():
    # (5,4,3)·(4,3,6) contracting axes ([1,2],[0,1]) -> (5,6); same as the einsum.
    A = fnp.asarray(np.random.rand(5, 4, 3))
    B = fnp.asarray(np.random.rand(4, 3, 6))
    td = cost(lambda: fnp.tensordot(A, B, axes=([1, 2], [0, 1])))
    es = cost(lambda: fnp.einsum("abc,bcf->af", A, B))
    assert td == es and td > A.size * B.size // 12  # not the old multiply-only count


def test_multi_dot_promotes_1d_operands():
    v = fnp.asarray(np.random.rand(64))
    M = fnp.asarray(np.random.rand(64, 64))
    w = fnp.asarray(np.random.rand(64))
    # v·M·w is a matvec chain: honest ~ 2*64*64 + 2*64, not 2*64^3.
    c = cost(lambda: fnp.linalg.multi_dot([v, M, w]))
    assert c < 2 * 64 * 64 * 64 // 10  # nowhere near the 1-D-as-k overcount


def test_polymul_equals_convolve():
    a = fnp.asarray(np.random.rand(500))
    b = fnp.asarray(np.random.rand(400))
    assert cost(lambda: fnp.polymul(a, b)) == cost(lambda: fnp.convolve(a, b))
