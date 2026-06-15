"""Public names that exist in flopscope but are server-side / in-process only.

These are NOT available in the flopscope client (the remote/grader env): cost
introspection, accounting/cache helpers, and result/AST type classes whose
client equivalent is ``RemoteArray``. Synced to the client by
``scripts/sync_client.py``; consulted by the client ``__getattr__`` paths and
the API-parity guard.
"""

from __future__ import annotations

SERVER_ONLY: frozenset[str] = frozenset(
    {
        # result / AST types (client uses RemoteArray + .symmetry metadata)
        "FlopscopeArray",
        "SymmetricTensor",
        "PathInfo",
        "StepInfo",
        # accounting / cache tooling
        "budget_reset",
        "namespace",
        "clear_einsum_cache",
        "einsum_cache_info",
        # flops.* cost-introspection helpers are added in Task C4
    }
)
