import pytest


def test_fnp_blacklisted_gives_clear_error():
    import flopscope.numpy as fnp

    with pytest.raises(AttributeError, match="not supported in the flopscope client"):
        _ = fnp.ndindex


def test_fnp_server_only_gives_clear_error():
    import flopscope.numpy as fnp

    with pytest.raises(AttributeError, match="server-side"):
        _ = fnp.SymmetricTensor


def test_fnp_real_op_still_resolves():
    import flopscope.numpy as fnp

    assert callable(fnp.matmul)  # real proxy, unaffected


def test_fnp_private_name_raises_plain_attributeerror():
    import flopscope.numpy as fnp

    with pytest.raises(AttributeError):
        _ = fnp._nonexistent_private  # underscore names must not be re-routed
