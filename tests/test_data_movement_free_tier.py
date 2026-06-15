"""Regression tests for the data-movement free-tier cost-model change.

See .aicrowd/superpowers/specs/2026-06-15-data-movement-free-tier-design.md.
"""
import pytest

import flopscope as flops
import flopscope.numpy as fnp
from flopscope._weights import get_weight, load_weights


# Ops that must bill 0 FLOPs under production weights (data movement / select).
FREE_DATA_MOVEMENT_OPS = [
    "hstack", "vstack", "column_stack", "dstack", "concatenate", "stack",
    "block", "bmat", "tile", "repeat", "resize", "pad", "roll", "tril",
    "triu", "insert", "append", "delete", "copyto", "diag", "diagflat",
    "meshgrid", "fromiter", "compress", "full", "full_like", "take",
    "take_along_axis", "put", "put_along_axis", "choose", "place", "putmask",
    "select", "extract", "fill_diagonal", "trim_zeros", "unstack",
]


@pytest.fixture
def production_weights(monkeypatch):
    """Load the packaged production weight table for this test only.

    conftest's autouse fixture resets to unit weights around every test.
    """
    monkeypatch.delenv("FLOPSCOPE_WEIGHTS_FILE", raising=False)
    load_weights()
    yield


@pytest.mark.parametrize("op", FREE_DATA_MOVEMENT_OPS)
def test_data_movement_op_is_weight_zero(production_weights, op):
    assert get_weight(op) == 0.0, f"{op} should be free (weight 0.0)"


def _mk1d():
    return fnp.asarray([float(i) for i in range(100)])


def _mk2d():
    return fnp.asarray([float(i) for i in range(100)]).reshape(10, 10)


# (build, call): build runs BEFORE n0 so ONLY the op under test is measured.
# (After migration, building inputs via reshape would itself add a record, so
# shaped inputs are constructed outside the measured region.)
VIEW_OPS_126 = {
    "reshape": (_mk1d, lambda a: fnp.reshape(a, (10, 10))),
    "transpose": (_mk2d, lambda a: fnp.transpose(a)),
    "swapaxes": (_mk2d, lambda a: fnp.swapaxes(a, 0, 1)),
    "moveaxis": (_mk2d, lambda a: fnp.moveaxis(a, 0, 1)),
    "squeeze": (lambda: _mk1d().reshape(1, 100), lambda a: fnp.squeeze(a)),
    "expand_dims": (_mk1d, lambda a: fnp.expand_dims(a, 0)),
    "copy": (_mk1d, lambda a: fnp.copy(a)),
    "flip": (_mk1d, lambda a: fnp.flip(a)),
    "fliplr": (_mk2d, lambda a: fnp.fliplr(a)),
    "flipud": (_mk2d, lambda a: fnp.flipud(a)),
    "rot90": (_mk2d, lambda a: fnp.rot90(a)),
    "atleast_1d": (_mk1d, lambda a: fnp.atleast_1d(a)),
    "atleast_2d": (_mk1d, lambda a: fnp.atleast_2d(a)),
    "atleast_3d": (_mk1d, lambda a: fnp.atleast_3d(a)),
    "fft.fftshift": (_mk1d, lambda a: fnp.fft.fftshift(a)),
    "fft.ifftshift": (_mk1d, lambda a: fnp.fft.ifftshift(a)),
    "hsplit": (_mk2d, lambda a: fnp.hsplit(a, 2)),
}


@pytest.mark.parametrize("name", sorted(VIEW_OPS_126))
def test_view_op_is_time_accounted(name):
    """#126: free view ops route through deduct -> >=1 op-log record, all 0 FLOPs."""
    build, call = VIEW_OPS_126[name]
    with flops.BudgetContext(flop_budget=10**9, quiet=True) as ctx:
        a = build()
        n0 = len(ctx.op_log)
        call(a)
        new = ctx.op_log[n0:]
    assert len(new) >= 1, f"{name}: no op-log record (still bypasses deduct)"
    assert all(r.flop_cost == 0 for r in new), f"{name}: free op billed nonzero FLOPs"
