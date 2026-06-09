"""Entry point for ``python -m flopscope_server``."""

from __future__ import annotations

import argparse
import os
import secrets
import sys

from flopscope_server._server import FlopscopeServer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Flopscope budget-controlled compute server",
    )
    parser.add_argument(
        "--url",
        default="ipc:///tmp/flopscope.sock",
        help="ZMQ endpoint to bind (default: ipc:///tmp/flopscope.sock)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Session inactivity timeout in seconds (default: 60.0)",
    )
    parser.add_argument(
        "--token-fd",
        type=int,
        default=None,
        help="If set, mint a random control token, write it (+newline) to this "
        "inherited fd, then close it. Shared only with the trusted parent.",
    )
    args = parser.parse_args()

    control_token = None
    if args.token_fd is not None:
        control_token = secrets.token_hex(32)
        os.write(args.token_fd, (control_token + "\n").encode("ascii"))
        os.close(args.token_fd)

    print(
        f"[flopscope-server] binding to {args.url}  (timeout={args.timeout}s)",
        file=sys.stderr,
    )

    server = FlopscopeServer(
        url=args.url, session_timeout_s=args.timeout, control_token=control_token
    )
    try:
        server.run()
    except KeyboardInterrupt:
        print("\n[flopscope-server] shutting down", file=sys.stderr)
        server.stop()


if __name__ == "__main__":
    main()
