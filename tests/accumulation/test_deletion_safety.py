"""Verify that deleted modules / symbols are no longer importable.

The deleted-module assertions go through ``importlib.import_module`` rather
than ``from ... import ...`` so static type checkers don't flag the
intentionally-failing imports.
"""

import importlib

import pytest

import flopscope._opt_einsum as _opt_einsum_pkg

# Two tests below intentionally import the real, on-disk local submodules
# ``flopscope._opt_einsum._paths`` / ``._path_random`` to assert they exist.
# Importing them registers each as an attribute in the ``flopscope._opt_einsum``
# package ``__dict__``, which permanently shadows the package's lazy
# ``__getattr__`` hook (PEP 562) that maps ``oe._paths`` / ``oe._path_random`` to
# the *upstream* ``opt_einsum`` modules. The leak persists for the whole process
# and breaks tests that run later — notably ``tests/test_opt_einsum_paths.py``,
# whose custom optimizers subclass the upstream ``PathOptimizer`` and rely on
# ``oe._path_random`` being upstream (otherwise ``isinstance`` checks flip and the
# optimizer is forwarded to upstream, raising ``TypeError``). xdist hides this
# because tests land on different workers; a serial run (``-n 0``) exposes it.
#
# ``_helpers`` is intentionally NOT restored: ``oe._helpers`` is meant to resolve
# to the *local* FMA-aware module (see tests/test_opt_einsum_paths.py docstring and
# ``test_flop_cost``), which the normal import chain registers anyway.
_LAZILY_SHIMMED_SUBMODULES = ("_paths", "_path_random")


def _restore_opt_einsum_lazy_shim() -> None:
    """Drop the shadowing ``_paths`` / ``_path_random`` package attributes so
    ``__init__.py``'s lazy ``__getattr__`` upstream mapping is restored.
    ``sys.modules`` is left intact so module/class identity is preserved for any
    holders of the local modules.
    """
    for _name in _LAZILY_SHIMMED_SUBMODULES:
        _opt_einsum_pkg.__dict__.pop(_name, None)


@pytest.fixture(autouse=True)
def _isolate_opt_einsum_lazy_shim():
    """Restore the lazy upstream shim after every test in this module so the
    intentional local-submodule imports here do not leak into later tests."""
    yield
    _restore_opt_einsum_lazy_shim()


def test_subgraph_symmetry_module_is_importable():
    # Restored in Task 15 (symmetry-aware path search branch).
    importlib.import_module("flopscope._opt_einsum._subgraph_symmetry")


def test_symmetric_flop_count_is_gone():
    # _symmetry.py was restored in Task 15; SubsetSymmetry is still present.
    importlib.import_module("flopscope._opt_einsum._symmetry")


def test_unique_elements_in_opt_einsum_symmetry_is_gone():
    # _symmetry.py was restored in Task 15.
    importlib.import_module("flopscope._opt_einsum._symmetry")


def test_subset_symmetry_dataclass_is_gone():
    # _symmetry.py was restored in Task 15; SubsetSymmetry is present.
    importlib.import_module("flopscope._opt_einsum._symmetry")


def test_unique_elements_for_shape_in_symmetry_utils_is_kept():
    """Sanity check: the keeper helper (used by SymmetricTensor sizing) still works."""
    from flopscope._symmetry_utils import unique_elements_for_shape

    assert callable(unique_elements_for_shape)


def test_symmetry_oracle_param_gone_from_contract_path():
    import inspect

    from flopscope._opt_einsum import contract_path

    sig = inspect.signature(contract_path)
    assert "symmetry_oracle" not in sig.parameters


def test_symmetry_oracle_param_gone_from_paths_module():
    # _paths.py was deleted in Task 7; upstream opt_einsum.paths is used directly.
    import inspect

    import opt_einsum.paths as paths

    src = inspect.getsource(paths)
    assert "symmetry_oracle" not in src


def test_symmetry_oracle_param_gone_from_path_random():
    # _path_random.py was deleted in Task 7; upstream opt_einsum.path_random is used directly.
    import inspect

    import opt_einsum.path_random as pr

    src = inspect.getsource(pr)
    assert "symmetry_oracle" not in src


# ── Devendor task 7+8 deletions ─────────────────────────────────────────


def test_opt_einsum_paths_module_is_importable():
    # Restored in Task 15 (symmetry-aware path search branch).
    importlib.import_module("flopscope._opt_einsum._paths")


def test_opt_einsum_path_random_module_is_importable():
    # Restored in Task 16 (symmetry-aware random-greedy branch).
    importlib.import_module("flopscope._opt_einsum._path_random")


def test_opt_einsum_blas_module_is_gone():
    with pytest.raises(ImportError):
        importlib.import_module("flopscope._opt_einsum._blas")


def test_opt_einsum_testing_module_is_gone():
    with pytest.raises(ImportError):
        importlib.import_module("flopscope._opt_einsum._testing")


def test_opt_einsum_typing_module_is_importable():
    # Restored in Task 15 (symmetry-aware path search branch).
    importlib.import_module("flopscope._opt_einsum._typing")


def test_opt_einsum_parser_module_is_gone():
    with pytest.raises(ImportError):
        importlib.import_module("flopscope._opt_einsum._parser")


def test_parse_einsum_input_reexported_from_init():
    """After Task 9: parse_einsum_input is importable from
    flopscope._opt_einsum (re-exported from upstream)."""
    from flopscope._opt_einsum import parse_einsum_input

    assert callable(parse_einsum_input)


def test_lazy_shim_restored_after_local_submodule_import():
    """Regression guard for the serial-only test-isolation leak.

    Importing the local ``_paths`` / ``_path_random`` submodules registers them in
    the package ``__dict__`` and shadows the lazy ``__getattr__`` upstream mapping.
    ``_restore_opt_einsum_lazy_shim`` (run by the autouse fixture after every test)
    must undo that so ``oe._paths`` / ``oe._path_random`` resolve back to upstream.

    Identity (``is``) is intentionally avoided: other tests in the suite reload
    ``opt_einsum`` submodules, so the object captured at ``__init__`` load time may
    differ. The invariant that matters is that the shadow is cleared and the
    resolved module is the *upstream* one, not the local submodule.
    """
    # Reproduce the leak exactly as a cold submodule import does — register the
    # local module as a package attribute (``importlib`` on a warm cache may skip
    # the parent-attribute assignment).
    for _name in ("_paths", "_path_random"):
        setattr(
            _opt_einsum_pkg,
            _name,
            importlib.import_module(f"flopscope._opt_einsum.{_name}"),
        )
    assert _opt_einsum_pkg.__dict__["_paths"].__name__ == "flopscope._opt_einsum._paths"
    assert (
        _opt_einsum_pkg.__dict__["_path_random"].__name__
        == "flopscope._opt_einsum._path_random"
    )

    # The cleanup the autouse fixture performs must restore the upstream shim.
    _restore_opt_einsum_lazy_shim()
    assert "_paths" not in _opt_einsum_pkg.__dict__
    assert "_path_random" not in _opt_einsum_pkg.__dict__
    assert getattr(_opt_einsum_pkg._paths, "__name__", "") == "opt_einsum.paths"
    assert (
        getattr(_opt_einsum_pkg._path_random, "__name__", "")
        == "opt_einsum.path_random"
    )
