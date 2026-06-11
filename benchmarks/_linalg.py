"""Benchmark linear algebra operations."""

from __future__ import annotations

import statistics

from benchmarks._perf import measure_flops
from flopscope._flops import svd_cost
from flopscope.numpy.linalg import lstsq_cost, pinv_cost

LINALG_OPS: list[str] = [
    "linalg.cholesky",
    "linalg.qr",
    "linalg.eig",
    "linalg.eigh",
    "linalg.eigvals",
    "linalg.eigvalsh",
    "linalg.svd",
    "linalg.svdvals",
    "linalg.solve",
    "linalg.inv",
    "linalg.lstsq",
    "linalg.pinv",
    "linalg.det",
    "linalg.slogdet",
]

# Ops that need symmetric positive-definite matrices.
_SPD_OPS = {"linalg.cholesky", "linalg.eigh", "linalg.eigvalsh"}

_FORMULA_STRINGS: dict[str, str] = {
    "linalg.cholesky": "n^3/3",
    "linalg.qr": "2*(2mnk - 2k^3/3), k=min(m,n) (reduced/complete)",
    "linalg.eig": "25n^3 (provisional)",
    "linalg.eigh": "9n^3 (provisional)",
    "linalg.eigvals": "10n^3 (provisional)",
    "linalg.eigvalsh": "4n^3/3 (provisional)",
    "linalg.svd": "4a^2b+22b^3 (full U, default) or 6ab^2+20b^3 (thin); a=max(m,n), b=min(m,n)",
    "linalg.svdvals": "2ab^2+2b^3 (values only), a=max(m,n), b=min(m,n)",
    "linalg.solve": "2n^3/3 + 2n^2 (nrhs=1)",
    "linalg.inv": "2n^3",
    "linalg.lstsq": "composed: svd+matmuls (lstsq_cost)",
    "linalg.pinv": "composed: svd+reconstruction (pinv_cost)",
    "linalg.det": "2n^3/3 + n",
    "linalg.slogdet": "2n^3/3 + n",
}


def _analytical_cost(op_name: str, n: int) -> int:
    """Return the textbook FLOP count for *op_name* on an (n, n) matrix.

    Parameters
    ----------
    op_name : str
        Operation name (e.g. ``"linalg.cholesky"``).
    n : int
        Matrix dimension.

    Returns
    -------
    int
        Analytical FLOP count.
    """
    m = n  # square matrices
    short = op_name.split(".")[-1]
    # qr: mode="reduced" (default); k=min(m,n)=n for square
    k = min(m, n)
    qr_factor = 2 * m * n * k - 2 * k**3 // 3
    # lstsq benchmark: A is (n,n), b is 1D vector of length n
    # pinv benchmark: A is (n,n)
    costs: dict[str, int] = {
        "cholesky": n**3 // 3,
        "qr": 2 * qr_factor,
        "eig": 25 * n**3,
        "eigh": 9 * n**3,
        "eigvals": 10 * n**3,
        "eigvalsh": 4 * n**3 // 3,
        "svd": svd_cost(m, n, with_vectors=True),
        "svdvals": svd_cost(m, n, with_vectors=False),
        "solve": 2 * n**3 // 3 + 2 * n * n,
        "inv": 2 * n**3,
        "lstsq": lstsq_cost(m, n, b_cols=1, b_ndim=1),
        "pinv": pinv_cost(m, n),
        "det": 2 * n**3 // 3 + n,
        "slogdet": 2 * n**3 // 3 + n,
    }
    return costs[short]


def benchmark_linalg(
    n: int = 1024,
    dtype: str = "float64",
    repeats: int = 10,
) -> tuple[dict[str, float], dict[str, dict]]:
    """Benchmark linalg ops, returning raw measurement per analytical FLOP.

    In perf mode this is actual FP ops / analytical FLOPs (correction factor).
    In timing mode this is nanoseconds / analytical FLOPs (same units as
    pointwise — the runner normalizes against baseline to get relative weights).

    Parameters
    ----------
    n : int
        Matrix dimension (n x n).
    dtype : str
        NumPy dtype string.
    repeats : int
        Number of repetitions per measurement.

    Returns
    -------
    tuple[dict[str, float], dict[str, dict]]
        A pair of (alphas, details). ``alphas`` maps op name to median
        raw measurement per analytical FLOP. ``details`` maps op name to
        a dict of raw benchmark metadata.
    """
    results: dict[str, float] = {}
    details: dict[str, dict] = {}

    for op in LINALG_OPS:
        dist_values: list[float] = []
        dist_raw_totals: list[int] = []

        if op in _SPD_OPS:
            # SPD matrices: A@A.T + n*I
            setups = [
                (
                    f"import numpy as np; rng = np.random.default_rng(42); "
                    f"_A = rng.standard_normal(({n}, {n})).astype(np.{dtype}); "
                    f"A = _A @ _A.T + {n} * np.eye({n}, dtype=np.{dtype})"
                ),
                (
                    f"import numpy as np; rng = np.random.default_rng(42); "
                    f"_A = rng.uniform(0.1, 1.0, size=({n}, {n})).astype(np.{dtype}); "
                    f"A = _A @ _A.T + {n} * np.eye({n}, dtype=np.{dtype})"
                ),
                (
                    f"import numpy as np; rng = np.random.default_rng(42); "
                    f"_A = rng.standard_normal(({n}, {n})).astype(np.{dtype}); "
                    f"A = _A @ _A.T + {n * 100} * np.eye({n}, dtype=np.{dtype})"
                ),
            ]
        else:
            # General, well-conditioned, ill-conditioned
            setups = [
                (
                    f"import numpy as np; rng = np.random.default_rng(42); "
                    f"A = rng.standard_normal(({n}, {n})).astype(np.{dtype})"
                ),
                (
                    f"import numpy as np; rng = np.random.default_rng(42); "
                    f"A = rng.standard_normal(({n}, {n})).astype(np.{dtype}); "
                    f"A = A + {n} * np.eye({n}, dtype=np.{dtype})"
                ),
                (
                    f"import numpy as np; rng = np.random.default_rng(42); "
                    f"_u = rng.standard_normal(({n}, {n})).astype(np.{dtype}); "
                    f"_s = np.logspace(0, -10, {n}, dtype=np.{dtype}); "
                    f"A = _u * _s @ _u.T"
                ),
            ]

        # Build bench code
        if op == "linalg.solve":
            bench_suffix = f"; b = np.ones({n}, dtype=np.{dtype})"
            bench = "np.linalg.solve(A, b)"
        elif op == "linalg.lstsq":
            bench_suffix = f"; b = np.ones({n}, dtype=np.{dtype})"
            bench = "np.linalg.lstsq(A, b, rcond=None)"
        else:
            bench_suffix = ""
            bench = f"np.{op}(A)"

        analytical = _analytical_cost(op, n)

        for setup in setups:
            full_setup = setup + bench_suffix
            try:
                result = measure_flops(full_setup, bench, repeats=repeats)
            except RuntimeError:
                continue
            measured = result.total_flops / repeats
            dist_values.append(measured / analytical if analytical else 0.0)
            dist_raw_totals.append(result.total_flops)

        if dist_values:
            results[op] = statistics.median(dist_values)
            if op in ("linalg.solve", "linalg.lstsq"):
                bm_size = f"A: ({n},{n}), b: ({n},)"
            else:
                bm_size = f"A: ({n},{n})"
            details[op] = {
                "category": "counted_custom",
                "measurement_mode": "blas",
                "analytical_formula": _FORMULA_STRINGS.get(op, ""),
                "analytical_flops": analytical,
                "benchmark_size": bm_size,
                "bench_code": bench,
                "repeats": repeats,
                "perf_instructions_total": dist_raw_totals,
                "distribution_alphas": dist_values,
            }

    return results, details
