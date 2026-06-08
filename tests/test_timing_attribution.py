"""Timing-bucket attribution tests (issue: callback/data-movement misattribution)."""
import time

import numpy as np
import pytest

import flopscope as flops
from flopscope._budget import _call_user_code, _counted_wrapper, get_active_budget


def test_user_code_time_lands_in_residual_not_overhead():
    """Wall time spent in _call_user_code must bill to residual, not overhead."""

    @_counted_wrapper
    def fake_callback_op():
        budget = get_active_budget()
        _call_user_code(budget, time.sleep, 0.05)

    flops.budget_reset()
    with flops.BudgetContext(flop_budget=10**6, quiet=True) as b:
        fake_callback_op()
    s = b.summary_dict()
    assert s["residual_wall_time_s"] >= 0.03, s
    assert s["flopscope_overhead_time_s"] < 0.02, s
    assert s["wall_time_s"] == pytest.approx(
        s["flopscope_backend_time_s"]
        + s["flopscope_overhead_time_s"]
        + s["residual_wall_time_s"],
        abs=1e-6,
    )
