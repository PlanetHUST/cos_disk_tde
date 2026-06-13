"""Primary script for running the TDE-rate workflow."""

from src.tde_rate import *  # noqa: F403
from src.tde_rate import run_cli


if __name__ == "__main__":
    run_cli()
