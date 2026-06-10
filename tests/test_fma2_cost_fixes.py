"""Charged-vs-honest FMA=2 cost tests for the audit fixes (one PR).

Charged = flops_used inside a BudgetContext (cost = flop_cost * weight),
under autoloaded packaged weights. Each test asserts the honest FMA=2 count.
"""
from __future__ import annotations

import numpy as np

import flopscope as f
import flopscope.numpy as fnp
from flopscope._weights import load_weights


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


def test_geomspace_logspace_cost_broadcast_output_times_transcendental():
    load_weights()  # conftest resets weights; reload packaged weights for this test
    assert cost(lambda: fnp.geomspace(1.0, 1000.0, 50)) == 16 * 50
    assert cost(lambda: fnp.logspace(0.0, 3.0, 50)) == 16 * 50
    start = fnp.asarray(np.ones(100))
    stop = fnp.asarray(np.full(100, 1000.0))
    assert cost(lambda: fnp.geomspace(start, stop, 50)) == 16 * 50 * 100  # broadcast B=100


def test_polydiv_scales_with_quotient_length():
    u = fnp.asarray(np.random.rand(800)); v = fnp.asarray(np.random.rand(50))
    expected = 1 + (800 - 50 + 1) * (2 * 50 + 1)
    assert cost(lambda: fnp.polydiv(u, v)) == expected
    # short-quotient case: n1=55, n2=50 → Q=6, cost=607 < 55*50=2750 (old formula)
    u2 = fnp.asarray(np.random.rand(55)); v2 = fnp.asarray(np.random.rand(50))
    assert cost(lambda: fnp.polydiv(u2, v2)) < 55 * 50


def test_interp_adds_search_term():
    x = fnp.asarray(np.linspace(0, 10, 1000))
    xp = fnp.asarray(np.linspace(0, 10, 16)); fp = fnp.asarray(np.random.rand(16))
    # honest = 3*n (arithmetic) + n*ceil(log2(M)); M=16 -> log2=4
    assert cost(lambda: fnp.interp(x, xp, fp)) == 3 * 1000 + 1000 * 4


def test_cross_is_three_per_output():
    a = fnp.asarray(np.random.rand(1000, 3)); b = fnp.asarray(np.random.rand(1000, 3))
    assert cost(lambda: fnp.cross(a, b)) == 3 * a.size
