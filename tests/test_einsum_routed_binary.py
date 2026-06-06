"""Unit tests for the shared _einsum_routed_binary contraction-cost helper."""

from __future__ import annotations

import numpy as np

import flopscope
from flopscope import BudgetContext, SymmetricTensor, SymmetryGroup


def _cost(fn, *args, **kwargs):
    with BudgetContext(flop_budget=int(1e20)) as bc:
        fn(*args, **kwargs)
    return bc.flops_used


def test_helper_charges_accumulation_total_for_2d_matmul():
    from flopscope._pointwise import _einsum_routed_binary

    a = np.random.RandomState(0).rand(10, 10)
    b = np.random.RandomState(1).rand(10, 10)
    with BudgetContext(flop_budget=int(1e20)) as bc:
        out = _einsum_routed_binary(
            "matmul", np.matmul, "ij,jk->ik", a, b, errstate=True, nan_check=True
        )
    assert bc.flops_used == 1900  # 2*10*10*10 - 10*10 (FMA=2)
    np.testing.assert_allclose(np.asarray(out), a @ b)


def test_helper_preserves_AA_symmetry():
    from flopscope._pointwise import _einsum_routed_binary

    A = flopscope.symmetrize(
        np.random.RandomState(0).randn(8, 8),
        symmetry=SymmetryGroup.symmetric(axes=(0, 1)),
    )
    with BudgetContext(flop_budget=int(1e20)):
        out = _einsum_routed_binary(
            "matmul", np.matmul, "ij,jk->ik", A, A, errstate=True, nan_check=True
        )
    assert isinstance(out, SymmetricTensor)
