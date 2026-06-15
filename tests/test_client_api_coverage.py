import importlib.util
import sys
from pathlib import Path

from flopscope._registry import REGISTRY

_BUCKET2 = [
    "ndindex", "nditer", "ndenumerate", "broadcast",
    "errstate", "printoptions", "get_printoptions", "set_printoptions",
    "finfo", "iinfo",
]


def test_bucket2_numpy_utils_are_blacklisted():
    for name in _BUCKET2:
        assert REGISTRY.get(name, {}).get("category") == "blacklisted", name


def test_server_only_set_syncs_to_client():
    from flopscope._server_only import SERVER_ONLY as core_set

    client_path = (
        Path(__file__).resolve().parent.parent
        / "flopscope-client"
        / "src"
        / "flopscope"
        / "_server_only_data.py"
    )
    assert client_path.exists(), "client _server_only_data.py missing (run sync_client.py)"
    spec = importlib.util.spec_from_file_location("_client_server_only_data", client_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_client_server_only_data"] = mod
    spec.loader.exec_module(mod)
    assert set(core_set) == set(mod.SERVER_ONLY)
    assert "SymmetricTensor" in core_set  # representative member
