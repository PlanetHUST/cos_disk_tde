"""Backward-compatible script for running the TDE-rate workflow.

The implementation lives in :mod:`src.tde_rate`. New code should import from
that package; this module remains available for existing scripts.
"""

from src.tde_rate import *  # noqa: F403
from src.tde_rate import run_cli


if __name__ == "__main__":
    run_cli()
