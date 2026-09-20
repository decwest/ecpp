"""``ecpp-paper``: reproduce the simulation tests of the ECPP paper.

Sub-commands follow the paper's numbering::

    ecpp-paper figure1   Fig. 1 (PP versus ECPP at two lookahead distances)
    ecpp-paper test1     Test 1: (omega_n, zeta, L_d) parameter sweep
    ecpp-paper test2     Test 2: preview range versus local response
    ecpp-paper test3     Test 3: phase-plane sweep of initial conditions
    ecpp-paper test4     Test 4: method comparison at representative conditions
    ecpp-paper all       everything above, in order
    ecpp-paper hw-reference   simulated references for real-robot Experiment 1

Every sub-command accepts ``--out-root DIR`` (default: the current directory)
and writes below ``DIR/generated_preview/``; ``--apply`` writes below
``DIR/generated/`` instead.  Pass ``-h`` after a sub-command for its options.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

COMMANDS = ("figure1", "test1", "test2", "test3", "test4", "all", "hw-reference")


def _run(command: str, remainder: list[str]) -> None:
    if command == "figure1":
        from .figure1 import main as run
    elif command == "test1":
        from .ieee_access import main_test1 as run
    elif command == "test2":
        from .test2_preview import main as run
    elif command == "test3":
        from .test3_phase_plane import main as run
    elif command == "test4":
        from .ieee_access import main_test4 as run
    elif command == "hw-reference":
        from .ieee_access import main_hw_reference as run
    else:
        raise ValueError(command)
    run(remainder)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="ecpp-paper",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("command", nargs="?", choices=COMMANDS,
                        help="which paper artifact to generate")
    args, remainder = parser.parse_known_args(argv)
    if args.command is None:
        parser.print_help()
        return
    if args.command == "all":
        shared = argparse.ArgumentParser(prog="ecpp-paper all")
        shared.add_argument("--out-root", default=None)
        shared.add_argument("--apply", action="store_true")
        shared.add_argument("--threads", type=int, default=None,
                            help="OpenMP threads for the Test 3 kernel")
        opts = shared.parse_args(remainder)
        common = (["--out-root", opts.out_root] if opts.out_root else []) + (["--apply"] if opts.apply else [])
        for command in ("figure1", "test1", "test2", "test3", "test4"):
            print(f"\n===== ecpp-paper {command} =====", flush=True)
            extra = ["--threads", str(opts.threads)] if (command == "test3" and opts.threads) else []
            _run(command, common + extra)
        return
    _run(args.command, list(remainder))


if __name__ == "__main__":
    main()
