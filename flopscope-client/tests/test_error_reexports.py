"""The client re-exports the participant-facing error/warning classes at the
top level (parity with in-process flopscope), so ``except flops.TimeExhaustedError``
written against the in-process docs also works on the client.
"""

from __future__ import annotations

import flopscope
import flopscope.errors as _errors

# Classes documented for participants that must be reachable as ``flops.<name>``
_REEXPORTED = [
    "BudgetExhaustedError",
    "TimeExhaustedError",
    "NoBudgetContextError",
    "UnsupportedFunctionError",
    "UnsupportedReturnType",
    "SymmetryError",
    "SymmetryLossWarning",
    "FlopscopeError",
    "FlopscopeWarning",
]


def test_error_classes_reexported_at_top_level():
    for name in _REEXPORTED:
        assert hasattr(flopscope, name), f"flopscope.{name} should be re-exported"
        assert getattr(flopscope, name) is getattr(_errors, name)


def test_reexported_errors_subclass_flopscope_error():
    for name in [
        "TimeExhaustedError",
        "UnsupportedFunctionError",
        "UnsupportedReturnType",
    ]:
        assert issubclass(getattr(flopscope, name), flopscope.FlopscopeError)
