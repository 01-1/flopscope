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
