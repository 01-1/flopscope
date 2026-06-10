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


def test_average_weighted_doubles_unweighted():
    W = fnp.asarray(np.random.rand(1000, 1000))
    w = fnp.asarray(np.random.rand(1000) + 0.5)
    unweighted = cost(lambda: fnp.average(W, axis=1))
    weighted = cost(lambda: fnp.average(W, axis=1, weights=w))
    assert weighted == unweighted + W.size      # +1 multiply pass for a*w


def test_var_is_four_passes_and_shape_stable():
    v = fnp.asarray(np.random.rand(1_000_000))
    assert cost(lambda: fnp.var(v)) == 4_000_000           # 4N, full reduction
    assert cost(lambda: fnp.std(v)) == 4_000_001           # 4N + 1 sqrt
    a = fnp.asarray(np.random.rand(500_000, 2))
    assert cost(lambda: fnp.var(a, axis=1)) == 4_000_000   # shape-stable, not 2(N-M)


def test_nanvar_matches_var_cost():
    v = fnp.asarray(np.random.rand(1_000_000))
    assert cost(lambda: fnp.nanvar(v)) == cost(lambda: fnp.var(v))


def test_trapezoid_is_four_per_element():
    y = fnp.asarray(np.random.rand(1_000_000))
    assert cost(lambda: fnp.trapezoid(y)) == 4 * y.size


def test_linspace_costs_broadcast_output():
    assert cost(lambda: fnp.linspace(0.0, 1.0, 50)) == 2 * 50
    start = fnp.asarray(np.zeros(100))
    stop = fnp.asarray(np.ones(100))
    assert cost(lambda: fnp.linspace(start, stop, 50)) == 2 * 50 * 100  # broadcast B=100
