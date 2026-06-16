"""End-to-end production-weight billing tests.

Most of the cost suite runs under UNIT weights: conftest's autouse
``reset_global_budget`` fixture calls ``reset_weights()``, clearing the table so
every op falls back to weight 1.0 — which pins each op's raw ``flop_cost``.
Weight TIERS are checked separately by ``test_weight_tier_policy.py`` (only that
each weight is a legal tier value). Neither pins what production actually
*bills* — ``flop_cost x weight`` — which is what participants are charged.

This module closes that gap: it loads the packaged production weights
(``data/default_weights.json``) and pins the billed cost for one representative
op per weight tier {0, 1, 8, 16}. A silent weight regression (e.g. a
transcendental sampler dropping from 16x to 1x) or a tier mislabel now fails
here — not only in the (unenforced) ``docs/reference/cost-model.md`` table.

Note: the former 4.0 "gather" tier (take/put/take_along_axis/put_along_axis etc.)
was replaced by the data-movement free tier (weight=0.0) in the cost-model
data-movement-free-tier change.
"""

import numpy as np
import pytest

import flopscope.numpy as fnp
from flopscope._budget import BudgetContext
from flopscope._weights import get_weight, load_weights, reset_weights

# Inputs are built once at import, OUTSIDE any BudgetContext, so only the
# measured op bills. (Under unit weights, array creation has its own non-zero
# flop_cost — see test_polynomial.py::test_polyfit_flopscope_array_inputs.)
_RNG = np.random.default_rng(0)
_A = fnp.asarray(_RNG.standard_normal(100))
_B = fnp.asarray(_RNG.standard_normal(100))
_IDX = fnp.asarray(_RNG.integers(0, 100, 100))


def _billed(call):
    with BudgetContext(flop_budget=10**12, quiet=True) as budget:
        call()
    return budget.flops_used


# label, weight_key, call, expected_billed (= flop_cost x weight), expected_weight.
# One op per tier; expected_billed verified against the live model at 100 elems.
# Tiers: {0, 1, 8, 16}. The old 4.0 gather tier is gone (data-movement free).
_TIER_CASES = [
    ("free: reshape", "reshape", lambda: fnp.reshape(_A, (10, 10)), 0, 0.0),
    ("free: take (was gather)", "take", lambda: fnp.take(_A, _IDX), 0, 0.0),
    ("arithmetic: add", "add", lambda: fnp.add(_A, _B), 100, 1.0),
    ("half: hanning", "hanning", lambda: fnp.hanning(100), 1600, 8.0),
    ("transcendental: exp", "exp", lambda: fnp.exp(_A), 1600, 16.0),
    (
        "transcendental: random.randn",
        "random.randn",
        lambda: fnp.random.randn(100),
        1600,
        16.0,
    ),
]


@pytest.fixture
def production_weights(monkeypatch):
    """Load the packaged production weight table for the test body.

    The autouse ``reset_global_budget`` fixture clears weights (-> unit 1.0)
    around every test; this loads the real ``default_weights.json`` for this
    test only. ``FLOPSCOPE_WEIGHTS_FILE`` is cleared first so the packaged
    default is used regardless of the environment.
    """
    monkeypatch.delenv("FLOPSCOPE_WEIGHTS_FILE", raising=False)
    load_weights()
    yield


@pytest.mark.parametrize(
    "label, weight_key, call, expected_billed, expected_weight", _TIER_CASES
)
def test_production_billed_cost_per_tier(
    production_weights, label, weight_key, call, expected_billed, expected_weight
):
    # The op carries its documented tier weight ...
    assert get_weight(weight_key) == expected_weight, (
        f"{label}: weight is {get_weight(weight_key)}, expected tier {expected_weight}"
    )
    # ... and production bills flop_cost x weight (what participants are charged).
    assert _billed(call) == expected_billed, label


def test_production_billed_equals_unit_flop_cost_times_weight(monkeypatch):
    """Invariant: production billed cost == raw flop_cost x weight, end to end."""
    monkeypatch.delenv("FLOPSCOPE_WEIGHTS_FILE", raising=False)
    for label, weight_key, call, _, _ in _TIER_CASES:
        reset_weights()  # unit weights -> raw flop_cost
        flop_cost = _billed(call)
        load_weights()  # production weights -> billed
        weight = get_weight(weight_key)
        assert _billed(call) == flop_cost * weight, (
            f"{label}: billed != flop_cost {flop_cost} x weight {weight}"
        )
    reset_weights()
