"""Billing + correctness tests for the counted symmetric ops."""

import numpy as np
import pytest  # noqa: F401  (used by parametrized tests added in later tasks)

import flopscope as flops
import flopscope.numpy as fnp  # noqa: F401  (used by tests added in later tasks)
from flopscope._symmetric import _check_generators, _project_core


def _billed(fn):
    """Return FLOPs charged by calling fn() inside a fresh BudgetContext."""
    with flops.BudgetContext(flop_budget=10**12):
        before = flops.budget_summary_dict()["flops_used"]
        fn()
        return flops.budget_summary_dict()["flops_used"] - before


def test_project_core_matches_reynolds_average():
    G = flops.SymmetryGroup.symmetric(axes=(0, 1))
    a = np.arange(16.0).reshape(4, 4)
    out = _project_core(a, G)
    expected = (a + a.T) / 2.0
    np.testing.assert_allclose(np.asarray(out), expected)


def test_check_generators_true_and_false():
    G = flops.SymmetryGroup.symmetric(axes=(0, 1))
    sym = np.array([[1.0, 2.0], [2.0, 3.0]])
    asym = np.array([[1.0, 2.0], [9.0, 3.0]])
    assert _check_generators(sym, G, atol=1e-6, rtol=1e-5) is True
    assert _check_generators(asym, G, atol=1e-6, rtol=1e-5) is False


@pytest.mark.parametrize("axes,G", [((0, 1), 2), ((0, 1, 2), 6)])
def test_symmetrize_bills_G_plus_1_times_n(axes, G):
    grp = flops.SymmetryGroup.symmetric(axes=axes)
    shape = (4,) * len(axes)
    n = int(np.prod(shape))
    with flops.BudgetContext(flop_budget=10**12):
        data = fnp.random.randn(*shape)
        b0 = flops.budget_summary_dict()["flops_used"]  # exclude sampling cost
        flops.symmetrize(data, symmetry=grp)
        delta = flops.budget_summary_dict()["flops_used"] - b0
    assert delta == max((G + 1) * n, 1)


def test_symmetrize_result_is_symmetric():
    grp = flops.SymmetryGroup.symmetric(axes=(0, 1))
    with flops.BudgetContext(flop_budget=10**12):
        data = fnp.random.randn(4, 4)
        s = flops.symmetrize(data, symmetry=grp)
        assert s.is_symmetric(symmetry=grp)
