"""Contraction-path search uses the FMA=2 accumulation cost (not legacy FMA=1).

These tests pin the billed total of multi-operand (>=3-operand) einsums to the
FMA=2 pairwise sum of the cheapest contraction order, and confirm binary
einsums (a single contraction step, no path choice) are unaffected.

FMA=2 per-step matmul cost for ``(m,p) @ (p,n) -> (m,n)`` is
``2*m*p*n - m*n`` (multiplies + adds, with the off-by-one accumulator-init
credit).
"""

import numpy as np

import flopscope.numpy as fnp
from flopscope import BudgetContext

rng = np.random.default_rng(0)


def _cost(subs, *arrs):
    with BudgetContext(flop_budget=int(1e20)) as bc:
        fnp.einsum(subs, *arrs)
    return bc.flops_used


def test_three_operand_chain_total_is_fma2_cheapest_path():
    A, B, C = (
        fnp.asarray(rng.standard_normal(s)) for s in [(10, 20), (20, 30), (30, 5)]
    )
    total = _cost("ij,jk,kl->il", A, B, C)
    abc = (2 * 10 * 20 * 30 - 10 * 30) + (2 * 10 * 30 * 5 - 10 * 5)  # (A@B)@C
    acb = (2 * 20 * 30 * 5 - 20 * 5) + (2 * 10 * 20 * 5 - 10 * 5)  # A@(B@C)
    assert total == min(abc, acb)


def test_binary_einsum_unaffected():
    A, B = (fnp.asarray(rng.standard_normal(s)) for s in [(10, 20), (20, 30)])
    assert _cost("ij,jk->ik", A, B) == 2 * 10 * 20 * 30 - 10 * 30


def test_three_operand_chain_strongly_prefers_one_order():
    # ab,bc,cd->ad with shapes chosen so the two association orders differ by
    # ~8x. (A@B)@C is cheap, A@(B@C) is expensive; the cheapest must win.
    A, B, C = (fnp.asarray(rng.standard_normal(s)) for s in [(5, 40), (40, 5), (5, 40)])
    total = _cost("ab,bc,cd->ad", A, B, C)
    o1 = (2 * 5 * 40 * 5 - 5 * 5) + (2 * 5 * 5 * 40 - 5 * 40)  # (A@B)@C
    o2 = (2 * 40 * 5 * 40 - 40 * 40) + (2 * 5 * 40 * 40 - 5 * 40)  # A@(B@C)
    assert o1 < o2  # sanity: shapes really do discriminate
    assert total == min(o1, o2)


def test_four_operand_chain_total_is_fma2_cheapest_path():
    # Brute-force the cheapest binary contraction tree under FMA=2 and assert
    # the billed total matches it.
    shapes = {"i": 8, "j": 16, "k": 24, "l": 6, "m": 30}
    A = fnp.asarray(rng.standard_normal((shapes["i"], shapes["j"])))
    B = fnp.asarray(rng.standard_normal((shapes["j"], shapes["k"])))
    C = fnp.asarray(rng.standard_normal((shapes["k"], shapes["l"])))
    D = fnp.asarray(rng.standard_normal((shapes["l"], shapes["m"])))
    total = _cost("ij,jk,kl,lm->im", A, B, C, D)

    # Reference: min over all binary trees of summed FMA=2 pairwise costs.
    # Each operand is a frozenset of labels; pairwise cost of contracting two
    # operands keeps labels that survive (appear elsewhere or in the output).
    output = frozenset("im")

    def pair_cost(s1, s2, others):
        either = s1 | s2
        keep = output | frozenset().union(*others) if others else output
        out = either & keep
        # full index space = product of all labels in `either`
        full = 1
        for c in either:
            full *= shapes[c]
        out_size = 1
        for c in out:
            out_size *= shapes[c]
        # FMA=2 single matmul-style step: 2*full - out_size when there is an
        # inner (contracted) dim, else just `full` (no reduction).
        if either - out:
            return 2 * full - out_size, out
        return full, out

    operands = [frozenset("ij"), frozenset("jk"), frozenset("kl"), frozenset("lm")]
    best = {"c": float("inf")}

    def search(ops, acc):
        if len(ops) == 1:
            best["c"] = min(best["c"], acc)
            return
        for a in range(len(ops)):
            for b in range(a + 1, len(ops)):
                rest = [ops[x] for x in range(len(ops)) if x not in (a, b)]
                c, merged = pair_cost(ops[a], ops[b], rest)
                search(rest + [merged], acc + c)

    search(operands, 0)
    assert total == best["c"]
