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
    "select", "extract", "fill_diagonal", "trim_zeros",
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
