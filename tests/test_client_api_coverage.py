import importlib.util
import sys
from pathlib import Path

from flopscope._registry import REGISTRY

_BUCKET2 = [
    "ndindex",
    "nditer",
    "ndenumerate",
    "broadcast",
    "errstate",
    "printoptions",
    "get_printoptions",
    "set_printoptions",
    "finfo",
    "iinfo",
]


def test_bucket2_numpy_utils_are_blacklisted():
    for name in _BUCKET2:
        assert REGISTRY.get(name, {}).get("category") == "blacklisted", name


_FLOPS_COST_HELPERS = [
    "flops.bartlett_cost",
    "flops.blackman_cost",
    "flops.cholesky_cost",
    "flops.cond_cost",
    "flops.det_cost",
    "flops.eig_cost",
    "flops.eigh_cost",
    "flops.eigvals_cost",
    "flops.eigvalsh_cost",
    "flops.fft_cost",
    "flops.fftn_cost",
    "flops.hamming_cost",
    "flops.hanning_cost",
    "flops.hfft_cost",
    "flops.inv_cost",
    "flops.kaiser_cost",
    "flops.lstsq_cost",
    "flops.matrix_norm_cost",
    "flops.matrix_power_cost",
    "flops.matrix_rank_cost",
    "flops.multi_dot_cost",
    "flops.norm_cost",
    "flops.pinv_cost",
    "flops.poly_cost",
    "flops.polyadd_cost",
    "flops.polyder_cost",
    "flops.polydiv_cost",
    "flops.polyfit_cost",
    "flops.polyint_cost",
    "flops.polymul_cost",
    "flops.polysub_cost",
    "flops.polyval_cost",
    "flops.qr_cost",
    "flops.rfft_cost",
    "flops.rfftn_cost",
    "flops.roots_cost",
    "flops.slogdet_cost",
    "flops.solve_cost",
    "flops.svdvals_cost",
    "flops.tensorinv_cost",
    "flops.tensorsolve_cost",
    "flops.trace_cost",
    "flops.unwrap_cost",
    "flops.vector_norm_cost",
]


def test_all_flops_cost_helpers_are_server_only():
    from flopscope._server_only import SERVER_ONLY

    missing = [n for n in _FLOPS_COST_HELPERS if n not in SERVER_ONLY]
    assert not missing, missing


def test_server_only_set_syncs_to_client():
    from flopscope._server_only import SERVER_ONLY as core_set

    client_path = (
        Path(__file__).resolve().parent.parent
        / "flopscope-client"
        / "src"
        / "flopscope"
        / "_server_only_data.py"
    )
    assert client_path.exists(), (
        "client _server_only_data.py missing (run sync_client.py)"
    )
    spec = importlib.util.spec_from_file_location(
        "_client_server_only_data", client_path
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_client_server_only_data"] = mod
    spec.loader.exec_module(mod)
    assert set(core_set) == set(mod.SERVER_ONLY)
    assert "SymmetricTensor" in core_set  # representative member
