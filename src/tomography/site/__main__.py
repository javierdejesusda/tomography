"""Command line entry: `python -m tomography.site build --out _site`."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from tomography.site import build as build_module


def main(argv: Sequence[str] | None = None) -> int:
    """Runs the site builder.

    Args:
        argv: Arguments; defaults to `sys.argv[1:]`.

    Returns:
        The process exit code.
    """
    parser = argparse.ArgumentParser(prog="python -m tomography.site")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="build the static site")
    build.add_argument("--results", type=Path, default=Path("results"))
    build.add_argument("--out", type=Path, default=Path("_site"))
    args = parser.parse_args(argv)
    doc = build_module.build(args.results, args.out)
    print(
        f"built {len(doc.figures) + len(doc.appendix)} figures and "
        f"{len(doc.tables) + len(doc.appendix_tables)} tables into {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
