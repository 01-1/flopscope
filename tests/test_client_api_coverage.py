from flopscope._registry import REGISTRY

_BUCKET2 = [
    "ndindex", "nditer", "ndenumerate", "broadcast",
    "errstate", "printoptions", "get_printoptions", "set_printoptions",
    "finfo", "iinfo",
]


def test_bucket2_numpy_utils_are_blacklisted():
    for name in _BUCKET2:
        assert REGISTRY.get(name, {}).get("category") == "blacklisted", name
