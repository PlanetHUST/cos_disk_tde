"""Supported command-line entry point for the TDE rate simulation.

The authoritative implementation currently lives in :mod:`test` for backward
compatibility with existing scripts. This module re-exports that public API so
``main.py`` and ``test.py`` cannot drift into different simulation versions.
"""

from test import *  # noqa: F403
from test import run_cli


if __name__ == "__main__":
    run_cli()
