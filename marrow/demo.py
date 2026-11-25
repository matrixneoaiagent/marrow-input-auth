"""Convenience entry point: ``python -m marrow.demo``."""

import sys

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main(["demo", *sys.argv[1:]]))
