"""Timing-bucket attribution tests (issue: callback/data-movement misattribution)."""
import time

import pytest

import flopscope as flops
import flopscope.numpy as fnp
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


def test_user_code_nested_flopscope_op_not_double_counted():
    """Callback that runs a real flopscope op: the op's time stays in
    backend/overhead, and the pure-Python remainder (sleep) goes to residual."""

    @_counted_wrapper
    def fake_callback_with_inner_op():
        budget = get_active_budget()

        def cb():
            time.sleep(0.03)
            fnp.add(fnp.array([1.0, 2.0]), fnp.array([3.0, 4.0]))
            return 0.0

        _call_user_code(budget, cb)

    flops.budget_reset()
    with flops.BudgetContext(flop_budget=10**6, quiet=True) as b:
        fake_callback_with_inner_op()
    s = b.summary_dict()
    # the inner flopscope op ran and was counted (nested > 0 path exercised)
    assert any(rec.op_name == "add" for rec in b.op_log), [r.op_name for r in b.op_log]
    # the 0.03s sleep (pure user time) lands in residual, not overhead
    assert s["residual_wall_time_s"] >= 0.02, s
    assert s["flopscope_overhead_time_s"] < 0.02, s
    # decomposition identity holds
    assert s["wall_time_s"] == pytest.approx(
        s["flopscope_backend_time_s"]
        + s["flopscope_overhead_time_s"]
        + s["residual_wall_time_s"],
        abs=1e-6,
    )
