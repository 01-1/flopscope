"""Regression guard: every `send_recv` call site in the client must sit inside a
dispatch span, so its transport + local marshaling is attributed to flopscope
overhead (dispatch) and never billed to the participant's residual.

A `<expr>.send_recv(...)` call is "span-covered" when ANY of:
  1. its nearest enclosing function is decorated `@timed_dispatch`, OR
  2. the call is lexically inside a `with dispatch_span():` block, OR
  3. its nearest enclosing function is wrapped via `timed_dispatch(<fn>)` in the
     same module (the `return timed_dispatch(proxy)` factory pattern).

Sibling of tests/test_no_bare_numpy_in_counted_ops.py in the core package — same
AST-guard idea, applied to the client/server boundary instead of in-process numpy.
"""

import ast
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src" / "flopscope"


def _attach_parents(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child._parent = node  # type: ignore[attr-defined]


def _is_timed_dispatch_decorated(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for d in fn.decorator_list:
        if isinstance(d, ast.Name) and d.id == "timed_dispatch":
            return True
        if isinstance(d, ast.Attribute) and d.attr == "timed_dispatch":
            return True
    return False


# NOTE: file-scoped (not closure-scoped). If one file ever had two different
# inner functions sharing a name where only one is wrapped, both would pass.
# Fine for the current codebase, where each file uses the name exactly once.
def _timed_dispatch_wrapped_names(tree: ast.AST) -> set:
    """Names X appearing as `timed_dispatch(X)` — the proxy-factory pattern."""
    names = set()
    for n in ast.walk(tree):
        if (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "timed_dispatch"
            and n.args
            and isinstance(n.args[0], ast.Name)
        ):
            names.add(n.args[0].id)
    return names


def _is_dispatch_span_with(node: ast.AST) -> bool:
    if not isinstance(node, ast.With):
        return False
    for item in node.items:
        ctx = item.context_expr
        if isinstance(ctx, ast.Call):
            f = ctx.func
            if isinstance(f, ast.Name) and f.id == "dispatch_span":
                return True
            if isinstance(f, ast.Attribute) and f.attr == "dispatch_span":
                return True
    return False


def _ancestors(node: ast.AST):
    cur = getattr(node, "_parent", None)
    while cur is not None:
        yield cur
        cur = getattr(cur, "_parent", None)


def _send_recv_calls(tree: ast.AST):
    for n in ast.walk(tree):
        if (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "send_recv"
        ):
            yield n


def test_no_bare_send_recv_outside_dispatch_span():
    offenders = []
    for path in sorted(_SRC.rglob("*.py")):
        tree = ast.parse(path.read_text())
        _attach_parents(tree)
        wrapped = _timed_dispatch_wrapped_names(tree)
        for call in _send_recv_calls(tree):
            covered = False
            enclosing = None
            for anc in _ancestors(call):
                if _is_dispatch_span_with(anc):
                    covered = True
                    break
                if isinstance(anc, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    enclosing = anc
                    covered = _is_timed_dispatch_decorated(anc) or anc.name in wrapped
                    break
            if not covered:
                rel = path.relative_to(_SRC.parent.parent)
                where = enclosing.name if enclosing else "<module>"
                offenders.append(f"{rel}:{call.lineno} in {where}()")
    assert not offenders, (
        "send_recv() call(s) outside a dispatch span — their wall time leaks "
        "into the participant's billed residual. Decorate the enclosing function "
        "with @timed_dispatch, or wrap the call in `with dispatch_span():`.\n"
        + "\n".join(offenders)
    )
