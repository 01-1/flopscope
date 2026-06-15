import pytest


def test_top_level_server_only_clear_error():
    import flopscope as flops

    with pytest.raises(AttributeError, match="server-side"):
        _ = flops.SymmetricTensor


def test_top_level_server_only_budget_reset():
    import flopscope as flops

    with pytest.raises(AttributeError, match="not available in the flopscope client"):
        _ = flops.budget_reset


def test_flops_submodule_has_getattr():
    import flopscope.flops as flops_mod

    # flops submodule now delegates unknown names through make_module_getattr,
    # so flops.* server-only names (populated in C4) raise the clear error
    # instead of a bare AttributeError.
    assert callable(getattr(flops_mod, "__getattr__", None))
