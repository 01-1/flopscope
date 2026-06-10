"""Tests for symmetry-aware linalg operations."""

import numpy

import flopscope.numpy.linalg as la
from flopscope import SymmetryGroup
from flopscope._budget import BudgetContext
from flopscope._symmetric import SymmetricTensor, as_symmetric


class TestEighValidation:
    def test_eigh_accepts_symmetric_tensor(self):
        A = numpy.array([[2.0, 1.0], [1.0, 3.0]])
        S = as_symmetric(A, symmetry=SymmetryGroup.symmetric(axes=(0, 1)))
        with BudgetContext(flop_budget=10**6, quiet=True):
            vals, vecs = la.eigh(S)
            assert vals.shape == (2,)

    def test_eigh_returns_plain_arrays(self):
        A = numpy.array([[2.0, 1.0], [1.0, 3.0]])
        S = as_symmetric(A, symmetry=SymmetryGroup.symmetric(axes=(0, 1)))
        with BudgetContext(flop_budget=10**6, quiet=True):
            vals, vecs = la.eigh(S)
            assert not isinstance(vals, SymmetricTensor)
            assert not isinstance(vecs, SymmetricTensor)


class TestSolveSymmetric:
    def test_solve_symmetric_uses_cubic_cost(self):
        n = 10
        A = numpy.eye(n) * 2.0
        S = as_symmetric(A, symmetry=SymmetryGroup.symmetric(axes=(0, 1)))
        b = numpy.ones(n)
        with BudgetContext(flop_budget=10**8, quiet=True) as budget:
            la.solve(S, b)
            # solve_cost(10, nrhs=1): 2*10^3//3 + 2*10^2 = 666 + 200 = 866
            assert budget.flops_used == 866

    def test_solve_plain_uses_cubic_cost(self):
        n = 10
        A = numpy.eye(n) * 2.0
        b = numpy.ones(n)
        with BudgetContext(flop_budget=10**8, quiet=True) as budget:
            la.solve(A, b)
            # solve_cost(10, nrhs=1): 2*10^3//3 + 2*10^2 = 666 + 200 = 866
            assert budget.flops_used == 866

    def test_solve_returns_plain(self):
        A = numpy.eye(3) * 2.0
        S = as_symmetric(A, symmetry=SymmetryGroup.symmetric(axes=(0, 1)))
        b = numpy.ones(3)
        with BudgetContext(flop_budget=10**6, quiet=True):
            result = la.solve(S, b)
            assert not isinstance(result, SymmetricTensor)


class TestDetSymmetric:
    def test_det_symmetric_uses_cubic_cost(self):
        n = 10
        A = numpy.eye(n) * 2.0
        S = as_symmetric(A, symmetry=SymmetryGroup.symmetric(axes=(0, 1)))
        with BudgetContext(flop_budget=10**8, quiet=True) as budget:
            la.det(S)
            # det_cost(10) = 2*10^3//3 + 10 = 666 + 10 = 676
            assert budget.flops_used == 2 * n**3 // 3 + n


class TestInvSymmetric:
    def test_inv_symmetric_returns_symmetric(self):
        A = numpy.eye(3) * 2.0
        S = as_symmetric(A, symmetry=SymmetryGroup.symmetric(axes=(0, 1)))
        with BudgetContext(flop_budget=10**6, quiet=True):
            result = la.inv(S)
            assert isinstance(result, SymmetricTensor)

    def test_inv_symmetric_cost(self):
        n = 10
        A = numpy.eye(n) * 2.0
        S = as_symmetric(A, symmetry=SymmetryGroup.symmetric(axes=(0, 1)))
        with BudgetContext(flop_budget=10**8, quiet=True) as budget:
            la.inv(S)
            expected = n**3 // 3 + n**3
            assert budget.flops_used == expected
