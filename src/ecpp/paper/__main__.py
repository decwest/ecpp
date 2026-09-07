"""Command-line entry point for canonical ECPP paper artifact generators."""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ecpp-paper")
    parser.add_argument(
        "command",
        nargs="?",
        choices=("figure1", "ieee-access", "test3-preview"),
        help="artifact generator to run",
    )
    args, remainder = parser.parse_known_args(argv)
    if args.command is None:
        parser.print_help()
        return
    if args.command == "figure1":
        from .figure1 import main as figure1_main

        figure1_main(remainder)
        return
    if args.command == "test3-preview":
        from .test3_preview import main as test3_main

        test3_main(remainder)
        return
    from .ieee_access import main as ieee_access_main

    ieee_access_main(remainder)


if __name__ == "__main__":
    main()
