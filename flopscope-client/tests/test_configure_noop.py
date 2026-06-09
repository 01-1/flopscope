"""flops.configure() warns that it is a no-op on flopscope-client.

The client ships ``_config.py`` verbatim, so ``configure`` stores settings but
they do not affect the remote (client/server) evaluation. The warning tells
participants their config will not carry into a graded submission. No live
server is needed — ``configure`` is purely local.
"""

from __future__ import annotations

import pytest

import flopscope as fnp
from flopscope.errors import ConfigureNoOpWarning


def test_configure_warns_noop_on_client():
    with pytest.warns(ConfigureNoOpWarning):
        fnp.configure(symmetry_warnings=True)


def test_configure_noop_warning_is_a_flopscope_warning():
    assert issubclass(ConfigureNoOpWarning, fnp.FlopscopeWarning)
