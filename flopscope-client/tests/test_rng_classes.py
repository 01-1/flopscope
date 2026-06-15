from flopscope._remote_array import (
    RemoteRandomState,
    RemoteSeedSequence,
    _encode_arg,
)


def test_remote_randomstate_encodes_as_rs_handle():
    rs = RemoteRandomState.__new__(RemoteRandomState)  # bypass dispatch __init__
    rs._handle_id = "g7"
    assert _encode_arg(rs) == {"__rs__": "g7"}


def test_remote_seedsequence_encodes_as_seq_handle():
    seq = RemoteSeedSequence.__new__(RemoteSeedSequence)
    seq._handle_id = "g9"
    assert _encode_arg(seq) == {"__seq__": "g9"}


def test_rng_class_names_resolve():
    from flopscope._remote_array import (
        RemoteGenerator,
        RemoteRandomState,
        RemoteSeedSequence,
    )

    import flopscope.numpy as fnp  # noqa: E402

    assert fnp.random.Generator is RemoteGenerator
    assert fnp.random.RandomState is RemoteRandomState
    assert fnp.random.SeedSequence is RemoteSeedSequence
