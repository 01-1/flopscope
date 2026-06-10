"""Charged-vs-honest assertions for the cost-constant unification (spec
2026-06-10). conftest resets weights to 1.0, so charged == flop_cost here;
packaged-table tests call load_weights() explicitly.

B.2 iterative constants (eig/svd families) are PROVISIONAL pending the
Plan-2 evidence pass; B.1 direct-solver constants are final.
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


# ---------------- Task 1: SVD family ----------------

def test_svdvals_values_only_constant():
    A = fnp.asarray(np.random.rand(200, 20))
    # a=max, b=min: 2*a*b^2 + 2*b^3 = 2*200*400 + 2*8000 = 176000
    assert cost(lambda: fnp.linalg.svdvals(A)) == 176_000


def test_svd_with_vectors_constant():
    A = fnp.asarray(np.random.rand(200, 20))
    # 6*a*b^2 + 20*b^3 = 480000 + 160000 = 640000
    assert cost(lambda: fnp.linalg.svd(A)) == 640_000
    # values-only path matches svdvals
    assert cost(lambda: fnp.linalg.svd(A, compute_uv=False)) == 176_000


def test_norm2_equals_svdvals_charge():
    A = fnp.asarray(np.random.rand(200, 20))
    assert cost(lambda: fnp.linalg.norm(A, 2)) == cost(lambda: fnp.linalg.svdvals(A))


def test_matrix_rank_is_values_svd_plus_threshold():
    A = fnp.asarray(np.random.rand(200, 20))
    assert cost(lambda: fnp.linalg.matrix_rank(A)) == 176_000 + 20


def test_cond_2norm_and_lu_paths():
    A = fnp.asarray(np.random.rand(200, 20))
    assert cost(lambda: fnp.linalg.cond(A)) == 176_000 + 1
    B = fnp.asarray(np.random.rand(100, 100) + 100 * np.eye(100))
    # p=1: inv-based 2n^3 + 4n^2 + 1
    assert cost(lambda: fnp.linalg.cond(B, 1)) == 2_000_000 + 40_000 + 1


def test_pinv_lstsq_self_correct_no_double_count():
    A = fnp.asarray(np.random.rand(200, 20))
    b = fnp.asarray(np.random.rand(200))
    from flopscope.numpy.linalg import pinv_cost, lstsq_cost
    # charged == raw composed flop_cost (weight must be 1.0 / absent)
    assert cost(lambda: fnp.linalg.pinv(A)) == pinv_cost(200, 20)
    assert cost(lambda: fnp.linalg.lstsq(A, b, rcond=None)) == lstsq_cost(200, 20, 1, 1)
    # and the composed values now use the with_vectors SVD constant:
    # pinv: 640000 + 20 + 400 + matmul(20,20,200)=156000 -> 796420
    assert pinv_cost(200, 20) == 796_420
    # lstsq: 640000 + matmul(20,200,1)=7980 + 20 + matmul(20,20,1)=780 -> 648780
    assert lstsq_cost(200, 20, 1, 1) == 648_780


def test_svd_family_packaged_weights_are_unity():
    load_weights()  # packaged table; linalg weights must now be 1.0
    A = fnp.asarray(np.random.rand(200, 20))
    assert cost(lambda: fnp.linalg.svdvals(A)) == 176_000


# ---------------- Task 2: direct solvers ----------------

def test_solve_is_lu_plus_triangular_and_nrhs_aware():
    A = fnp.asarray(np.random.rand(100, 100) + 100 * np.eye(100))
    b1 = fnp.asarray(np.random.rand(100))
    b8 = fnp.asarray(np.random.rand(100, 8))
    third = 2 * 100**3 // 3                      # 666666
    assert cost(lambda: fnp.linalg.solve(A, b1)) == third + 2 * 100**2 * 1
    assert cost(lambda: fnp.linalg.solve(A, b8)) == third + 2 * 100**2 * 8


def test_inv_constants():
    A = fnp.asarray(np.random.rand(100, 100) + 100 * np.eye(100))
    assert cost(lambda: fnp.linalg.inv(A)) == 2 * 100**3


def test_tensorsolve_tensorinv_reduce_to_solve_inv():
    a = fnp.asarray(np.random.rand(8, 3, 24))
    b = fnp.asarray(np.random.rand(8, 3))
    # n = prod(trailing) = 24: 2*24^3//3 + 2*24^2 = 9216 + 1152
    assert cost(lambda: fnp.linalg.tensorsolve(a, b)) == 9216 + 1152
    ai = fnp.asarray(np.random.rand(4, 6, 4, 6))
    # n = prod(leading 2) = 24: 2*24^3 = 27648
    assert cost(lambda: fnp.linalg.tensorinv(ai)) == 27_648


# ---------------- Task 3: cholesky / qr / det ----------------

def test_cholesky_third_cubed():
    A = fnp.asarray(np.random.rand(100, 100))
    SPD = fnp.asarray(np.asarray(A) @ np.asarray(A).T + 100 * np.eye(100))
    assert cost(lambda: fnp.linalg.cholesky(SPD)) == 100**3 // 3


def test_qr_mode_aware():
    A = fnp.asarray(np.random.rand(200, 50))
    factor = 2 * 200 * 50 * 50 - 2 * 50**3 // 3          # 916,667 (Householder)
    assert cost(lambda: fnp.linalg.qr(A, mode="r")) == factor
    assert cost(lambda: fnp.linalg.qr(A)) == 2 * factor   # reduced: + form Q
    S = fnp.asarray(np.random.rand(100, 100))
    fs = 2 * 100**3 - 2 * 100**3 // 3                     # 1,333,334
    assert cost(lambda: fnp.linalg.qr(S)) == 2 * fs


def test_det_slogdet_lu_based():
    A = fnp.asarray(np.random.rand(100, 100) + 100 * np.eye(100))
    expected = 2 * 100**3 // 3 + 100
    assert cost(lambda: fnp.linalg.det(A)) == expected
    assert cost(lambda: fnp.linalg.slogdet(A)) == expected


def test_direct_family_packaged_weights_are_unity():
    load_weights()
    A = fnp.asarray(np.random.rand(100, 100))
    SPD = fnp.asarray(np.asarray(A) @ np.asarray(A).T + 100 * np.eye(100))
    assert cost(lambda: fnp.linalg.cholesky(SPD)) == 100**3 // 3


# ---------------- Task 4: eigen family (PROVISIONAL constants) ----------------

def test_eigen_family_constants():
    n = 100
    G = fnp.asarray(np.random.rand(n, n))
    S = fnp.asarray(np.random.rand(n, n)); S = fnp.asarray(np.asarray(S) + np.asarray(S).T)
    assert cost(lambda: fnp.linalg.eig(G)) == 25 * n**3
    assert cost(lambda: fnp.linalg.eigvals(G)) == 10 * n**3
    assert cost(lambda: fnp.linalg.eigh(S)) == 9 * n**3
    assert cost(lambda: fnp.linalg.eigvalsh(S)) == 4 * n**3 // 3


def test_roots_composes_eigvals():
    p = fnp.asarray(np.random.rand(101))      # degree 100 -> 100 roots
    assert cost(lambda: fnp.roots(p)) == 10 * 100**3


def test_poly_2d_inherits_new_eigvals_constant():
    M = fnp.asarray(np.random.rand(50, 50))
    # poly 2-D = 2*n^2 + eigvals_cost(n) = 5000 + 10*125000
    assert cost(lambda: fnp.poly(M)) == 2 * 50 * 50 + 10 * 50**3
