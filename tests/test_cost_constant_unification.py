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
from flopscope._weights import load_weights, reset_weights


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
    # numpy default full_matrices=True, non-square (200,20):
    # 4*a^2*b + 22*b^3 = 4*200*200*20 + 22*20^3 = 3200000 + 176000 = 3376000
    assert cost(lambda: fnp.linalg.svd(A)) == 3_376_000
    # thin SVD: 6*a*b^2 + 20*b^3 = 6*200*400 + 20*8000 = 480000 + 160000 = 640000
    assert cost(lambda: fnp.linalg.svd(A, full_matrices=False)) == 640_000
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
    from flopscope.numpy.linalg import lstsq_cost, pinv_cost

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
    third = 2 * 100**3 // 3  # 666666
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
    factor = 2 * 200 * 50 * 50 - 2 * 50**3 // 3  # 916,667 (Householder)
    assert cost(lambda: fnp.linalg.qr(A, mode="r")) == factor
    assert cost(lambda: fnp.linalg.qr(A)) == 2 * factor  # reduced: + form Q
    S = fnp.asarray(np.random.rand(100, 100))
    fs = 2 * 100**3 - 2 * 100**3 // 3  # 1,333,334
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
    S = fnp.asarray(np.random.rand(n, n))
    S = fnp.asarray(np.asarray(S) + np.asarray(S).T)
    assert cost(lambda: fnp.linalg.eig(G)) == 25 * n**3
    assert cost(lambda: fnp.linalg.eigvals(G)) == 10 * n**3
    assert cost(lambda: fnp.linalg.eigh(S)) == 9 * n**3
    assert cost(lambda: fnp.linalg.eigvalsh(S)) == 4 * n**3 // 3


def test_roots_composes_eigvals():
    p = fnp.asarray(np.random.rand(101))  # degree 100 -> 100 roots
    assert cost(lambda: fnp.roots(p)) == 10 * 100**3


def test_poly_2d_inherits_new_eigvals_constant():
    M = fnp.asarray(np.random.rand(50, 50))
    # poly 2-D = 2*n^2 + eigvals_cost(n) = 5000 + 10*125000
    assert cost(lambda: fnp.poly(M)) == 2 * 50 * 50 + 10 * 50**3


# ---------------- Task 5: multivariate_normal ----------------


def test_multivariate_normal_bills_decomposition_and_transform():
    d, N = 50, 100
    mean, cov = np.zeros(d), np.eye(d)
    expected = d**3 // 3 + 2 * N * d * d + 16 * N * d  # 41666+500000+80000
    assert cost(lambda: fnp.random.multivariate_normal(mean, cov, size=N)) == expected


def test_multivariate_normal_default_size_is_one_sample():
    d = 30
    expected = d**3 // 3 + 2 * d * d + 16 * d
    assert (
        cost(lambda: fnp.random.multivariate_normal(np.zeros(d), np.eye(d))) == expected
    )


def test_multivariate_normal_packaged_weight_is_unity():
    load_weights()
    d = 30
    expected = d**3 // 3 + 2 * d * d + 16 * d
    assert (
        cost(lambda: fnp.random.multivariate_normal(np.zeros(d), np.eye(d))) == expected
    )


def test_generator_and_randomstate_mvn_match_module_path():
    d, N = 50, 100
    expected = d**3 // 3 + 2 * N * d * d + 16 * N * d

    # default_rng construction costs 0 FLOPs, so build inside cost() is fine.
    def gen():
        rng = fnp.random.default_rng(42)
        rng.multivariate_normal(np.zeros(d), np.eye(d), size=N)

    def rs():
        r = fnp.random.RandomState(42)
        r.multivariate_normal(np.zeros(d), np.eye(d), size=N)

    assert cost(gen) == expected
    assert cost(rs) == expected


def test_mvn_tuple_size_parity_across_paths():
    d = 20
    expected = d**3 // 3 + 2 * 20 * d * d + 16 * 20 * d  # N = 4*5 = 20
    mean, cov = np.zeros(d), np.eye(d)
    assert (
        cost(lambda: fnp.random.multivariate_normal(mean, cov, size=(4, 5))) == expected
    )

    def gen():
        fnp.random.default_rng(0).multivariate_normal(mean, cov, size=(4, 5))

    def rs():
        fnp.random.RandomState(0).multivariate_normal(mean, cov, size=(4, 5))

    assert cost(gen) == expected
    assert cost(rs) == expected


# ---------------- norm-family batch dims ----------------


def test_norm_family_bills_batch_dims():
    X = fnp.asarray(np.random.rand(100, 10, 10))
    x2 = fnp.asarray(np.random.rand(10, 10))
    v100 = fnp.asarray(np.random.rand(100, 10))
    v10 = fnp.asarray(np.random.rand(10))
    # batched charge == batch_size * single-slice charge
    assert cost(lambda: fnp.linalg.norm(X, "fro", axis=(-2, -1))) == 100 * cost(
        lambda: fnp.linalg.norm(x2, "fro")
    )
    assert cost(lambda: fnp.linalg.norm(X, 2, axis=(-2, -1))) == 100 * cost(
        lambda: fnp.linalg.norm(x2, 2)
    )
    assert cost(lambda: fnp.linalg.vector_norm(v100, axis=-1)) == 100 * cost(
        lambda: fnp.linalg.vector_norm(v10)
    )
    assert cost(lambda: fnp.linalg.matrix_norm(X)) == 100 * cost(
        lambda: fnp.linalg.matrix_norm(x2)
    )
    assert cost(lambda: fnp.linalg.matrix_norm(X, ord=2)) == 100 * cost(
        lambda: fnp.linalg.matrix_norm(x2, ord=2)
    )


def test_norm_family_unbatched_unchanged():
    x2 = fnp.asarray(np.random.rand(10, 10))
    v = fnp.asarray(np.random.rand(10))
    assert cost(lambda: fnp.linalg.norm(x2, "fro")) == 200  # 2*numel
    assert cost(lambda: fnp.linalg.norm(x2, 2)) == 4000  # values-SVD 10x10
    assert cost(lambda: fnp.linalg.vector_norm(v)) == 20  # 2*n
    assert cost(lambda: fnp.linalg.norm(v)) == 20  # 1-D path
    X = fnp.asarray(np.random.rand(100, 10, 10))
    assert (
        cost(lambda: fnp.linalg.norm(X)) == 2 * X.size
    )  # axis=None flattens: unchanged


# ---------------- generators: retstep/arange/indices (audit-2 verified) ----------------


def test_linspace_retstep_costs_full_grid():
    assert cost(lambda: fnp.linspace(0.0, 1.0, 50, retstep=True)) == 2 * 50
    start = fnp.asarray(np.zeros(100))
    stop = fnp.asarray(np.ones(100))
    assert cost(lambda: fnp.linspace(start, stop, 50, retstep=True)) == 2 * 50 * 100


def test_arange_two_flops_per_element():
    assert cost(lambda: fnp.arange(1000)) == 2 * 1000
    assert cost(lambda: fnp.arange(0)) == 0


def test_indices_sparse_and_dense():
    assert cost(lambda: fnp.indices((1000, 1000), sparse=True)) == 2000
    assert cost(lambda: fnp.indices((1000, 1000))) == 2_000_000


# ---------------- svd full_matrices + general-p norms (audit-2 verified) ----------------


def test_svd_full_matrices_default_costs_full_u():
    A = fnp.asarray(np.random.rand(200, 20))
    a, b = 200, 20
    assert (
        cost(lambda: fnp.linalg.svd(A)) == 4 * a * a * b + 22 * b**3
    )  # default full_matrices=True
    assert (
        cost(lambda: fnp.linalg.svd(A, full_matrices=False))
        == 6 * a * b * b + 20 * b**3
    )
    S = fnp.asarray(np.random.rand(50, 50))
    assert cost(lambda: fnp.linalg.svd(S)) == cost(
        lambda: fnp.linalg.svd(S, full_matrices=False)
    )  # square unchanged
    assert cost(lambda: fnp.linalg.svd(A, compute_uv=False)) == 2 * a * b * b + 2 * b**3


def test_vector_norm_general_p_bills_pow():
    v = fnp.asarray(np.random.rand(100))
    assert cost(lambda: fnp.linalg.vector_norm(v, ord=3)) == 18 * 100 + 16
    assert cost(lambda: fnp.linalg.vector_norm(v, ord=2)) == 2 * 100
    assert cost(lambda: fnp.linalg.norm(v, 3)) == 18 * 100 + 16
    V = fnp.asarray(np.random.rand(50, 100))
    assert cost(lambda: fnp.linalg.vector_norm(V, axis=-1, ord=3)) == 50 * (
        18 * 100 + 16
    )


# ---------------- lexsort / sort_complex / select (audit-2 verified) ----------------


def test_lexsort_bills_all_slices():
    from flopscope._flops import sort_cost

    k1 = fnp.asarray(np.random.rand(100, 70))  # axis=-1: 100 slices of n=70, 2 keys
    k2 = fnp.asarray(np.random.rand(100, 70))
    assert cost(lambda: fnp.lexsort((k1, k2), axis=-1)) == 2 * 100 * sort_cost(70)
    v1 = fnp.asarray(np.random.rand(1000))  # 1-D unchanged
    v2 = fnp.asarray(np.random.rand(1000))
    assert cost(lambda: fnp.lexsort((v1, v2))) == 2 * sort_cost(1000)


def test_sort_complex_per_slice():
    from flopscope._flops import sort_cost

    a = fnp.asarray(np.random.rand(100, 70) + 1j * np.random.rand(100, 70))
    assert cost(lambda: fnp.sort_complex(a)) == 100 * sort_cost(70)
    v = fnp.asarray(np.random.rand(1000) + 1j)
    assert cost(lambda: fnp.sort_complex(v)) == sort_cost(1000)


def test_select_bills_broadcast_output():
    x = fnp.asarray(np.random.rand(1000))
    conds = [np.asarray(x) < 0.3, np.asarray(x) > 0.7]
    # scalar choices used to collapse the charge; output is 1000 elements
    assert cost(lambda: fnp.select(conds, [0.0, 1.0], default=0.5)) == 1000


# ---------------- choice(p=) + diff prepend/append (audit-2 verified) ----------------


def test_choice_weighted_bills_cdf_build():
    from flopscope._flops import _ceil_log2

    n, draws = 1000, 10
    p = np.full(n, 1.0 / n)
    unweighted = cost(lambda: fnp.random.choice(n, size=draws))
    weighted = cost(lambda: fnp.random.choice(n, size=draws, p=p))
    assert weighted == unweighted + 3 * n + draws * _ceil_log2(n)
    g = cost(lambda: fnp.random.default_rng(0).choice(n, size=draws, p=p))
    r = cost(lambda: fnp.random.RandomState(0).choice(n, size=draws, p=p))
    assert g == weighted and r == weighted


def test_diff_bills_and_accepts_prepend_append():
    a = fnp.asarray(np.random.rand(1000))
    plain = cost(lambda: fnp.diff(a))  # 999
    pre = fnp.asarray(np.random.rand(5))
    padded = cost(lambda: fnp.diff(a, prepend=pre, append=0.0))  # L=1006 -> 1005
    assert plain == 999 and padded == 1005
    # crash regression: FlopscopeArray prepend must not raise
    with f.BudgetContext(flop_budget=10**9, quiet=True):
        out = np.asarray(fnp.diff(a, prepend=pre))
    np.testing.assert_array_equal(out, np.diff(np.asarray(a), prepend=np.asarray(pre)))


# ---------------- stats composites (audit-2 verified) ----------------


def test_stats_norm_family_composites():
    x = fnp.asarray(np.random.rand(1000) * 0.8 + 0.1)
    import flopscope.stats as fstats

    assert cost(lambda: fstats.norm.ppf(x)) == 83 * 1000
    assert cost(lambda: fstats.norm.pdf(x)) == 27 * 1000
    assert cost(lambda: fstats.norm.cdf(x)) == 48 * 1000


def test_stats_ppf_composites_packaged_weight_unity():
    load_weights()
    x = fnp.asarray(np.random.rand(100) * 0.8 + 0.1)
    import flopscope.stats as fstats

    assert cost(lambda: fstats.norm.ppf(x)) == 83 * 100
    assert cost(lambda: fstats.truncnorm.ppf(x, -1.0, 1.0)) == 81 * 100
    assert cost(lambda: fstats.lognorm.ppf(x, 0.5)) == 106 * 100


# ---- stats gap fixes (audit-2 verified, PR fix/cost-model-gaps) ----


def test_stats_laplace_cdf_composite():
    """laplace.cdf: two eager exp branches + arithmetic/select = 40/elem, weight 1.0."""
    import flopscope.stats as fstats

    x = fnp.asarray(np.linspace(-3.0, 3.0, 1000))
    assert cost(lambda: fstats.laplace.cdf(x)) == 40 * 1000


def test_stats_laplace_ppf_composite():
    """laplace.ppf: two eager log branches + edge selects = 51/elem, weight 1.0."""
    import flopscope.stats as fstats

    q = fnp.asarray(np.random.rand(1000) * 0.98 + 0.01)
    assert cost(lambda: fstats.laplace.ppf(q)) == 51 * 1000


def test_stats_lognorm_pdf_composite():
    """lognorm.pdf: log + exp + arithmetic = 62/elem (calibration 62.30), weight 1.0."""
    import flopscope.stats as fstats

    x = fnp.asarray(np.abs(np.random.rand(1000)) + 0.1)
    assert cost(lambda: fstats.lognorm.pdf(x, 0.5)) == 62 * 1000


def test_stats_lognorm_cdf_composite():
    """lognorm.cdf: log + erf rational approx + arithmetic = 70/elem, weight 1.0."""
    import flopscope.stats as fstats

    x = fnp.asarray(np.abs(np.random.rand(1000)) + 0.1)
    assert cost(lambda: fstats.lognorm.cdf(x, 0.5)) == 70 * 1000


# ---------------------------------------------------------------------------
# Audit gap fixes: copy/scatter/stack ops (13 ops)
# ---------------------------------------------------------------------------


def test_insert_bills_numel_output():
    a = np.arange(10000.0)
    # insert single element: output size = 10001
    assert cost(lambda: fnp.insert(a, 0, 1.0)) == 10001
    # scalar insert into 100x100 with axis: output = 10100
    m = np.ones((100, 100))
    assert cost(lambda: fnp.insert(m, 0, 7.0, axis=1)) == 10100
    # regression: must NOT scale with values.size alone
    assert cost(lambda: fnp.insert(np.zeros(1_000_000), 500000, 1.0)) == 1_000_001


def test_append_bills_numel_output():
    a = np.ones(10_000)
    # append one element: arr.size + values.size = 10001
    assert cost(lambda: fnp.append(a, [1.0])) == 10_001
    # append empty: still bills arr.size (materializes copy)
    assert cost(lambda: fnp.append(a, [])) == 10_000
    # family parity: append == concatenate for same shape
    v = np.ones(5_000)
    c_append = cost(lambda: fnp.append(a, v))
    c_concat = cost(lambda: fnp.concatenate([a, v]))
    assert c_append == c_concat == 15_000


def test_delete_bills_numel_output():
    a = np.arange(10000.0)
    # delete one element: output size = 9999
    assert cost(lambda: fnp.delete(a, 5)) == 9999
    # delete nothing: still bills numel(output) = 10000 (materializes copy)
    assert cost(lambda: fnp.delete(a, [])) == 10000
    # family parity: delete == concatenate for same shape
    c_delete = cost(lambda: fnp.delete(a, 5))
    c_concat = cost(lambda: fnp.concatenate([a[:5], a[6:]]))
    assert c_delete == c_concat == 9999


def test_copyto_bills_dst_numel():
    dst = np.zeros(10000)
    # scalar src: should bill dst.size = 10000
    assert cost(lambda: fnp.copyto(dst, 3.14)) == 10000
    # broadcast src row -> 100x100 dst
    dst2d = np.zeros((100, 100))
    assert cost(lambda: fnp.copyto(dst2d, np.arange(100.0))) == 10000
    # full-shape copy unchanged
    assert cost(lambda: fnp.copyto(dst, np.ones(10000))) == 10000
    # broadcast where mask: ones((100,1)) -> (100,100) = 10000 writes
    where_mask = np.ones((100, 1), dtype=bool)

    def _copyto_where():
        fnp.copyto(dst2d, np.ones((100, 100)), where=where_mask)  # type: ignore[arg-type]

    assert cost(_copyto_where) == 10000


def test_hstack_bills_numel_output():
    v = np.ones(100)
    w = np.ones(100)
    # 1-D hstack: output = 200
    assert cost(lambda: fnp.hstack([v, w])) == 200
    # parity with concatenate
    assert cost(lambda: fnp.hstack([v, w])) == cost(lambda: fnp.concatenate([v, w]))
    # 2-D hstack: two (3,4) -> (3,8) = 24
    A = np.ones((3, 4))
    assert cost(lambda: fnp.hstack([A, A])) == 24


def test_column_stack_bills_numel_output():
    # three 100-elem vectors -> (100, 3) = 300
    v = np.ones(100)
    assert cost(lambda: fnp.column_stack([v, v, v])) == 300
    # parity with stack(axis=1)
    assert cost(lambda: fnp.column_stack([v, v, v])) == cost(
        lambda: fnp.stack([v, v, v], axis=1)
    )
    # mixed 1-D/2-D: (50,2) + (50,) -> (50,3) = 150
    m = np.ones((50, 2))
    w = np.ones(50)
    assert cost(lambda: fnp.column_stack([m, w])) == 150


def test_row_stack_bills_numel_output():
    v = np.ones(100)
    w = np.ones(100)
    # row_stack == vstack: two (100,) -> (2, 100) = 200
    assert cost(lambda: fnp.row_stack([v, w])) == 200
    # exact parity with vstack
    assert cost(lambda: fnp.row_stack([v, w])) == cost(lambda: fnp.vstack([v, w]))


def test_tril_bills_numel_output():
    m = np.ones((100, 100))
    # weight from spec: 1.0 (materializing-copy tier per triu spec)
    assert cost(lambda: fnp.tril(m)) == 10_000
    # batch dims billed
    ms = np.ones((50, 100, 100))
    assert cost(lambda: fnp.tril(ms)) == 500_000


def test_triu_bills_numel_output():
    m = np.ones((100, 100))
    assert cost(lambda: fnp.triu(m)) == 10_000
    # batch dims billed
    ms = np.ones((50, 100, 100))
    assert cost(lambda: fnp.triu(ms)) == 500_000


def test_put_bills_numel_indices():
    # conftest resets weights to 1.0 so charged == flop_cost = numel(indices)
    a = np.zeros(10000)
    assert cost(lambda: fnp.put(a, np.arange(7), np.ones(7))) == 7
    # wrap mode: 1000 indices -> flop_cost = 1000
    assert cost(lambda: fnp.put(np.zeros(4), np.arange(1000), 1.0, mode="wrap")) == 1000
    # must NOT scale with destination size (was: a.size; now: ind.size)
    assert cost(lambda: fnp.put(np.zeros(10000), np.arange(7), np.ones(7))) == 7
    # With packaged weights loaded (weight=4.0): charged = int(7*4.0) = 28
    try:
        load_weights()
        assert cost(lambda: fnp.put(np.zeros(10000), np.arange(7), np.ones(7))) == 28
        assert (
            cost(lambda: fnp.put(np.zeros(4), np.arange(1000), 1.0, mode="wrap"))
            == 4000
        )
    finally:
        reset_weights()


def test_put_along_axis_bills_scattered_elements():
    # conftest resets weights to 1.0 so charged == flop_cost = scattered count
    dest = np.zeros(100)
    assert cost(lambda: fnp.put_along_axis(dest, np.arange(5), np.ones(5), 0)) == 5
    # dest=(100,10), indices=(1,5) -> scattered = (100*10 // 10) * 5 = 100*5 = 500
    dest2d = np.zeros((100, 10))
    assert (
        cost(
            lambda: fnp.put_along_axis(dest2d, np.zeros((1, 5), dtype=int), 1.0, axis=1)
        )
        == 500
    )
    # large J > M: flop_cost = J (axis 0: arr.size // arr.shape[0] == 1) * J = J
    dest_small = np.zeros(10)
    assert (
        cost(
            lambda: fnp.put_along_axis(
                dest_small, np.zeros(1_000_000, dtype=np.int64), 1.0, 0
            )
        )
        == 1_000_000
    )
    # With packaged weights loaded (weight=4.0): charged = scattered_count * 4
    try:
        load_weights()
        assert cost(lambda: fnp.put_along_axis(dest, np.arange(5), np.ones(5), 0)) == 20
        assert (
            cost(
                lambda: fnp.put_along_axis(
                    dest2d, np.zeros((1, 5), dtype=int), 1.0, axis=1
                )
            )
            == 2000
        )
        assert (
            cost(
                lambda: fnp.put_along_axis(
                    dest_small, np.zeros(1_000_000, dtype=np.int64), 1.0, 0
                )
            )
            == 4_000_000
        )
    finally:
        reset_weights()


def test_roll_bills_numel_output():
    a = np.zeros((100, 100))
    # single-axis roll: numel(output) * weight 1.0 = 10000
    assert cost(lambda: fnp.roll(a, 7)) == 10_000
    # multi-axis roll: same size output
    assert cost(lambda: fnp.roll(a, (3, 5), axis=(0, 1))) == 10_000


def test_meshgrid_sparse_and_copy():
    v = np.arange(10.0)
    # dense default: unchanged 200
    assert cost(lambda: fnp.meshgrid(v, v)) == 200
    # sparse=True: sum of input lengths = 20
    assert cost(lambda: fnp.meshgrid(v, v, sparse=True)) == 20
    # copy=False: views, floor = 1
    assert cost(lambda: fnp.meshgrid(v, v, copy=False)) == 1
    # sparse+copy=False: views, floor = 1
    assert cost(lambda: fnp.meshgrid(v, v, sparse=True, copy=False)) == 1
    # scale guard: sparse 1000x1000 = 2000, not 2,000,000
    big = np.arange(1000.0)
    assert cost(lambda: fnp.meshgrid(big, big, sparse=True)) == 2000


def test_stats_uniform_cdf_composite():
    """uniform.cdf: sub + div + 2 clip compare/selects = 4/elem, weight 1.0."""
    import flopscope.stats as fstats

    x = fnp.asarray(np.random.rand(1000))
    assert cost(lambda: fstats.uniform.cdf(x)) == 4 * 1000


def test_stats_cauchy_pdf_composite():
    """cauchy.pdf: pure-arithmetic z=(x-loc)/scale; 1/(pi*scale*(1+z^2)) = 6/elem, weight 1.0."""
    import flopscope.stats as fstats

    x = fnp.asarray(np.linspace(-3.0, 3.0, 1000))
    assert cost(lambda: fstats.cauchy.pdf(x)) == 6 * 1000


# ---------------------------------------------------------------------------
# Audit gap fixes: clip / count_nonzero / correlate / gradient / nanmean / nanmedian
# ---------------------------------------------------------------------------


def test_clip_two_bounds_bills_2x_numel():
    # clip with both bounds = 2 compare-selects/elem; numel=100 → 200
    v = fnp.asarray(np.random.rand(100))
    assert cost(lambda: fnp.clip(v, -1.0, 1.0)) == 200


def test_clip_one_bound_bills_numel():
    v = fnp.asarray(np.random.rand(100))
    # single-bound clip: 1 compare-select/elem; 100 → 100
    assert cost(lambda: fnp.clip(v, None, 1.0)) == 100
    assert cost(lambda: fnp.clip(v, -1.0, None)) == 100


def test_clip_broadcast_output_shape():
    # broadcast: a=(1,1), bounds=(500,500) → output numel=500*500; 2 bounds → 500000
    a = fnp.asarray(np.zeros((1, 1)))
    lo = fnp.asarray(-np.ones((500, 500)))
    hi = fnp.asarray(np.ones((500, 500)))
    assert cost(lambda: fnp.clip(a, lo, hi)) == 500_000


def test_clip_no_bound_bills_numel_floor():
    # no-bound clip: materializing copy floor → numel
    v = fnp.asarray(np.random.rand(100))
    assert cost(lambda: fnp.clip(v)) == 100


def test_clip_matches_minimum_maximum():
    # bit-exact equivalence: clip(-1,1) == minimum(maximum(v,-1),1) == 2*numel
    v = fnp.asarray(np.random.rand(100))
    clip_cost = cost(lambda: fnp.clip(v, -1.0, 1.0))
    composed_cost = cost(lambda: fnp.minimum(fnp.maximum(v, -1.0), 1.0))
    assert clip_cost == composed_cost == 200


def test_count_nonzero_bills_numel_axis_independent():
    # axis-independent: always charges numel(input)
    a = fnp.asarray(np.random.rand(2, 50))  # numel=100
    assert cost(lambda: fnp.count_nonzero(a, axis=0)) == 100  # was 50
    a2 = fnp.asarray(np.random.rand(1000, 2))
    assert cost(lambda: fnp.count_nonzero(a2, axis=1)) == 2000  # was 1000
    a3 = fnp.asarray(np.random.rand(4, 5, 6))
    assert cost(lambda: fnp.count_nonzero(a3, axis=(0, 2))) == 120  # was 115


def test_count_nonzero_full_reduction():
    # dedicated wrapper: numel(input)=100 (not numel-1=99 from _counted_reduction)
    a = fnp.asarray(np.random.rand(100))
    assert cost(lambda: fnp.count_nonzero(a)) == 100  # was 99


def test_correlate_valid_mode_honest():
    # valid (numpy default): honest = (2*min-1)*(max-min+1)
    # n=m=100: (2*100-1)*(100-100+1) = 199*1 = 199 (was 19800)
    a = fnp.asarray(np.random.rand(100))
    v = fnp.asarray(np.random.rand(100))
    assert cost(lambda: fnp.correlate(a, v)) == 199


def test_correlate_full_mode():
    a = fnp.asarray(np.random.rand(100))
    v = fnp.asarray(np.random.rand(100))
    # full: 2*100*100 - 100 - 100 + 1 = 19801 (was 19800, off-by-one)
    assert cost(lambda: fnp.correlate(a, v, mode="full")) == 19_801


def test_correlate_same_mode():
    a = fnp.asarray(np.random.rand(100))
    v = fnp.asarray(np.random.rand(100))
    # same, n=m=100: spec says 14900
    assert cost(lambda: fnp.correlate(a, v, mode="same")) == 14_900


def test_correlate_mode_int_and_case():
    a = fnp.asarray(np.random.rand(100))
    v = fnp.asarray(np.random.rand(100))
    # mode=0 == "valid", mode=2 == "full", "V" == "valid"
    assert cost(lambda: fnp.correlate(a, v, mode="valid")) == 199
    assert cost(lambda: fnp.correlate(a, v, mode="full")) == 19_801
    assert cost(lambda: fnp.correlate(a, v, mode="V")) == 199


def test_correlate_asymmetric_valid():
    # n=10000, m=100: valid honest = (2*100-1)*(10000-100+1) = 199*9901 = 1970299
    a = fnp.asarray(np.random.rand(10_000))
    v = fnp.asarray(np.random.rand(100))
    assert cost(lambda: fnp.correlate(a, v)) == 1_970_299


def test_correlate_scalar():
    a = fnp.asarray(np.random.rand(1))
    v = fnp.asarray(np.random.rand(1))
    assert cost(lambda: fnp.correlate(a, v)) == 1


def test_gradient_spacing_surcharge_arange():
    # np.arange(100.) passes the bit-exact uniformity test → only diff+equal+all-reduce
    # surcharge = 3*(L-1) = 3*99 = 297; base = 196; total = 493
    f = fnp.asarray(np.linspace(0, 1, 100) ** 2)
    x = fnp.asarray(np.arange(100.0))
    assert cost(lambda: fnp.gradient(f, x)) == 196 + 297


def test_gradient_spacing_surcharge_nonuniform():
    # non-uniform float spacing: full surcharge
    # 1-D L=100, S=100: 3*100*98//100 + 10*98 + 3*99 + 4*100//100
    #                  = 294 + 980 + 297 + 4 = 1575; total = 196 + 1575 = 1771
    rng = np.random.default_rng(0)
    f = fnp.asarray(rng.random(100))
    x = fnp.asarray(np.sort(rng.random(100)))
    assert cost(lambda: fnp.gradient(f, x)) == 1771


def test_gradient_uniform_scalar_unchanged():
    # uniform scalar spacing (no coord array): no surcharge; base unchanged
    f = fnp.asarray(np.linspace(0, 1, 100) ** 2)
    assert cost(lambda: fnp.gradient(f)) == 196
    assert cost(lambda: fnp.gradient(f, 0.5)) == 196


def test_nanmean_matches_mean():
    # spec: flop_cost(nanmean) == flop_cost(mean) for all shapes/axes
    a = fnp.asarray(np.random.rand(8, 5))
    assert cost(lambda: fnp.nanmean(a)) == cost(lambda: fnp.mean(a))
    assert cost(lambda: fnp.nanmean(a, axis=1)) == cost(lambda: fnp.mean(a, axis=1))
    a2 = fnp.asarray(np.random.rand(1000, 2))
    assert cost(lambda: fnp.nanmean(a2, axis=1)) == cost(lambda: fnp.mean(a2, axis=1))


def test_nanmean_full_reduction_100():
    # (10,10) full reduction: mean charges sum_cost(99) + 1 divide = 100
    a = fnp.asarray(np.random.rand(10, 10))
    assert cost(lambda: fnp.nanmean(a)) == 100  # was 99


def test_nanmedian_matches_median():
    # spec: flop_cost(nanmedian) == flop_cost(median) for all shapes/axes
    a = fnp.asarray(np.random.rand(8, 5))
    assert cost(lambda: fnp.nanmedian(a)) == cost(lambda: fnp.median(a))
    assert cost(lambda: fnp.nanmedian(a, axis=1)) == cost(lambda: fnp.median(a, axis=1))
    a2 = fnp.asarray(np.random.rand(1000, 2))
    assert cost(lambda: fnp.nanmedian(a2, axis=1)) == cost(
        lambda: fnp.median(a2, axis=1)
    )


def test_nanmedian_tier2_full_reduction():
    # (10,10) full reduction: Tier-2 = 1 orbit × 100 = 100 (was 99)
    a = fnp.asarray(np.random.rand(10, 10))
    assert cost(lambda: fnp.nanmedian(a)) == 100  # was 99


def test_stats_gap_fixes_packaged_weight_unity():
    """With packaged weights loaded, new composite constants must hold (weight=1.0)."""
    load_weights()
    import flopscope.stats as fstats

    x100 = fnp.asarray(np.linspace(-3.0, 3.0, 100))
    q100 = fnp.asarray(np.random.rand(100) * 0.98 + 0.01)
    xpos100 = fnp.asarray(np.abs(np.random.rand(100)) + 0.1)
    u100 = fnp.asarray(np.random.rand(100))

    assert cost(lambda: fstats.laplace.cdf(x100)) == 40 * 100
    assert cost(lambda: fstats.laplace.ppf(q100)) == 51 * 100
    assert cost(lambda: fstats.lognorm.pdf(xpos100, 0.5)) == 62 * 100
    assert cost(lambda: fstats.lognorm.cdf(xpos100, 0.5)) == 70 * 100
    assert cost(lambda: fstats.uniform.cdf(u100)) == 4 * 100
    assert cost(lambda: fstats.cauchy.pdf(x100)) == 6 * 100


# ---------------- reductions & predicates (audit-2 verified) ----------------


def test_nanpercentile_nanquantile_positional_q_and_cost():
    a = fnp.asarray(np.random.rand(500, 2))
    assert cost(lambda: fnp.nanpercentile(a, 50)) == cost(lambda: fnp.percentile(a, 50))
    assert cost(lambda: fnp.nanquantile(a, 0.5, axis=1)) == cost(
        lambda: fnp.quantile(a, 0.5, axis=1)
    )
    with f.BudgetContext(flop_budget=10**9, quiet=True):
        r = np.asarray(fnp.nanpercentile(a, 50, axis=1))
    np.testing.assert_allclose(r, np.nanpercentile(np.asarray(a), 50, axis=1))


def test_ptp_two_passes():
    v = fnp.asarray(np.random.rand(10_000))
    assert cost(lambda: fnp.ptp(v)) == 2 * 10_000 - 1  # 2*(N-1)+1
    A = fnp.asarray(np.random.rand(100, 50))
    assert cost(lambda: fnp.ptp(A, axis=1)) == 2 * (100 * 50 - 100) + 100


def test_average_matches_mean_and_bills_weight_pipeline():
    W = fnp.asarray(np.random.rand(1000, 1000))
    w = fnp.asarray(np.random.rand(1000) + 0.5)
    assert cost(lambda: fnp.average(W, axis=1)) == cost(lambda: fnp.mean(W, axis=1))
    weighted = cost(lambda: fnp.average(W, axis=1, weights=w))
    unweighted = cost(lambda: fnp.average(W, axis=1))
    # + a*w pass (numel) + w.sum reduction (numel - M) where M=1000
    assert weighted == unweighted + W.size + (W.size - 1000)


def test_dtype_predicates_are_free():
    v = fnp.asarray(np.random.rand(10_000))
    assert cost(lambda: fnp.iscomplexobj(v)) == 0
    assert cost(lambda: fnp.isrealobj(v)) == 0


# ---------------- audit-gap fixes (2026-06-11) ----------------


def test_trace_batch_multiply():
    """numpy.trace must multiply single-matrix diagonal by number of batch matrices."""
    # single matrix unchanged
    assert cost(lambda: fnp.trace(np.ones((10, 10)))) == 10
    # default axis1=0, axis2=1: matrix dims (100,10), n_traces=10 along dim2 -> 10*10=100
    assert cost(lambda: fnp.trace(np.ones((100, 10, 10)))) == 100
    # explicit axis1=1, axis2=2: matrix (10,10) in 100 batches -> 100*10=1000
    assert cost(lambda: fnp.trace(np.ones((100, 10, 10)), axis1=1, axis2=2)) == 1000
    # higher-rank batch: axis1=0,axis2=1 -> matrix (2,3)=min 2, n_traces=4*10*10=400 -> 800
    assert cost(lambda: fnp.trace(np.ones((2, 3, 4, 10, 10)))) == 2 * 400
    # zero-size matrix dim -> 0
    assert cost(lambda: fnp.trace(np.ones((5, 0, 10)))) == 0
    # zero along batch axes; axis1=0,axis2=1, shape (0,10,10) -> a.shape[0]=0, zero product
    assert cost(lambda: fnp.trace(np.ones((0, 10, 10)))) == 0


def test_allclose_6per_elem():
    """allclose must bill 7*numel(broadcast) - 1 (6/elem tolerance core + all-reduce)."""
    a = np.random.rand(100)
    b = np.random.rand(100)
    assert cost(lambda: fnp.allclose(a, b)) == 7 * 100 - 1
    # broadcast case
    a2 = np.random.rand(100, 1)
    b2 = np.random.rand(1, 100)
    assert cost(lambda: fnp.allclose(a2, b2)) == 7 * 10_000 - 1


def test_isclose_6per_elem():
    """isclose must bill 6*numel(output) (tolerance core: sub+2*abs+mul+add+cmp)."""
    a = np.random.rand(100)
    b = np.random.rand(100)
    assert cost(lambda: fnp.isclose(a, b)) == 6 * 100
    # broadcast
    a2 = np.random.rand(100, 1)
    b2 = np.random.rand(1, 100)
    assert cost(lambda: fnp.isclose(a2, b2)) == 6 * 10_000


def test_histogram_string_bins_charges_more_than_int():
    """histogram with string estimator bins must charge >= int-bins equivalent + 2n."""
    import math
    rng = np.random.default_rng(42)
    a = rng.standard_normal(1000)
    # 'auto' resolves to some nbins; must charge strictly more than int-path
    nb = len(np.histogram_bin_edges(a, 'auto')) - 1
    int_cost = cost(lambda: fnp.histogram(a, bins=nb))
    str_cost = cost(lambda: fnp.histogram(a, bins='auto'))
    assert str_cost >= int_cost + 2 * 1000, (
        f"string 'auto' cost {str_cost} not >= int cost {int_cost} + 2n=2000"
    )


def test_histogram_bin_edges_wrapped_bins_no_crash():
    """histogram_bin_edges with FlopscopeArray bins must not crash."""
    a = np.random.rand(100)
    edges = fnp.linspace(0.0, 1.0, 11)
    with f.BudgetContext(flop_budget=10**12, quiet=True) as b:
        result = fnp.histogram_bin_edges(a, bins=edges)
    plain = np.histogram_bin_edges(a, bins=np.linspace(0.0, 1.0, 11))
    np.testing.assert_array_equal(np.asarray(result), plain)


def test_histogram_wrapped_bins_no_crash():
    """histogram with FlopscopeArray bins must not crash."""
    a = np.random.rand(100)
    edges = fnp.linspace(0.0, 1.0, 11)
    with f.BudgetContext(flop_budget=10**12, quiet=True):
        counts, out_edges = fnp.histogram(a, bins=edges)
    plain_counts, plain_edges = np.histogram(a, bins=np.linspace(0.0, 1.0, 11))
    np.testing.assert_array_equal(np.asarray(counts), plain_counts)


def test_bartlett_4n():
    """bartlett must bill 4*n (compare+divide+add+select per sample, FMA=2)."""
    assert cost(lambda: fnp.bartlett(50)) == 4 * 50
    assert cost(lambda: fnp.bartlett(1)) == 4 * 1


def test_blackman_40n():
    """blackman must bill 40*n (2 cosine evals @16 + 8 arith per sample)."""
    assert cost(lambda: fnp.blackman(50)) == 40 * 50


def test_kaiser_23n():
    """kaiser must bill 23*n (1 Bessel I0 @16 + 7 arith per sample)."""
    assert cost(lambda: fnp.kaiser(50, 14.0)) == 23 * 50
    assert cost(lambda: fnp.kaiser(10, 5.0)) == 23 * 10


def test_hfft_half_cost():
    """fft.hfft must bill rfft_cost(n_out) not full complex cost."""
    import math
    # default n_out = 2*(n_in - 1) = 126 for input length 64
    a = np.random.rand(64).astype(complex)
    # rfft_cost(126) = 5 * (126//2) * ceil(log2(126)) = 5 * 63 * 7 = 2205
    expected = 5 * (126 // 2) * math.ceil(math.log2(126))
    assert cost(lambda: fnp.fft.hfft(a)) == expected
    # explicit n=200: rfft_cost(200) = 5 * 100 * ceil(log2(200)) = 5*100*8=4000
    expected2 = 5 * (200 // 2) * math.ceil(math.log2(200))
    assert cost(lambda: fnp.fft.hfft(a, n=200)) == expected2


def test_ihfft_rfft_cost():
    """fft.ihfft must bill rfft_cost(n) not full complex cost."""
    import math
    a = np.random.rand(64)
    # rfft_cost(64) = 5 * 32 * 6 = 960
    expected = 5 * (64 // 2) * math.ceil(math.log2(64))
    assert cost(lambda: fnp.fft.ihfft(a)) == expected
    # batched (8, 64): 8 * 960 = 7680
    batch = np.random.rand(8, 64)
    assert cost(lambda: fnp.fft.ihfft(batch)) == 8 * expected
