"""Command-line interface for the TDE-rate workflow."""

import argparse
import time
from pathlib import Path
from typing import Sequence

from .config import SECONDS_PER_MINUTE, TDEDynConfig
from .workflow import run_simulation


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Run the dynamical NSC-AGN TDE-rate simulation."
    )
    parser.add_argument(
        "num_samples",
        nargs="?",
        type=int,
        default=5,
        help="mass-grid samples per axis (default: 5)",
    )
    parser.add_argument(
        "num_repeats",
        nargs="?",
        type=int,
        default=3,
        help="Monte Carlo repeats per grid point (default: 3)",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        type=Path,
        default=Path("./tde_dyn_results"),
        help="output directory",
    )
    parser.add_argument(
        "damping_cpu_count",
        nargs="?",
        type=int,
        default=None,
        help="worker count; scheduler/local CPU count is used when omitted",
    )
    parser.add_argument("--duty-cycle-myr", type=float, default=10.0)
    parser.add_argument("--smbh-min-msun", type=float, default=1.0e3)
    parser.add_argument("--smbh-max-msun", type=float, default=1.0e5)
    parser.add_argument("--nsc-min-msun", type=float, default=1.0e5)
    parser.add_argument("--nsc-max-msun", type=float, default=1.0e7)
    parser.add_argument("--metallicity", type=float, default=0.01)
    parser.add_argument("--max-copies-per-object", type=int, default=30)
    parser.add_argument(
        "--no-resupply",
        action="store_true",
        help="disable relaxation-driven resupply copies",
    )
    parser.add_argument(
        "--resample-copy-orbits",
        action="store_true",
        help="sample new orbits instead of reusing the primary orbit",
    )
    parser.add_argument(
        "--direct-tde",
        action="store_true",
        help="count stars ending inside the configured TDE radius",
    )
    parser.add_argument("--tde-radius-rg", type=float, default=10.0)
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=5,
        help="save partial CSV results every N grid points",
    )
    parser.add_argument(
        "--no-visualizations",
        action="store_true",
        help="skip plot generation",
    )
    parser.add_argument(
        "--no-debug-results",
        action="store_true",
        help="suppress detailed per-realization counters",
    )
    return parser


def run_cli(argv: Sequence[str] | None = None) -> None:
    """Parse command-line arguments and run the simulation."""
    arguments = build_parser().parse_args(argv)
    config = TDEDynConfig(
        num_samp=arguments.num_samples,
        num_repeats=arguments.num_repeats,
        duty_cycle_myr=arguments.duty_cycle_myr,
        m_smbh_min_msun=arguments.smbh_min_msun,
        m_smbh_max_msun=arguments.smbh_max_msun,
        m_nsc_min_msun=arguments.nsc_min_msun,
        m_nsc_max_msun=arguments.nsc_max_msun,
        output_dir=arguments.output_dir,
        metallicity=arguments.metallicity,
        dyn_resupply=not arguments.no_resupply,
        max_copies_per_object=arguments.max_copies_per_object,
        reuse_same_orbit_for_copies=(not arguments.resample_copy_orbits),
        tde_radius_rg=arguments.tde_radius_rg,
        direct_tde=arguments.direct_tde,
        damping_cpu_count=arguments.damping_cpu_count,
        checkpoint_grid_interval=arguments.checkpoint_interval,
        create_visualizations=not arguments.no_visualizations,
        print_debug_results=not arguments.no_debug_results,
    )
    start_time_s = time.time()
    run_simulation(config)
    runtime_s = time.time() - start_time_s
    print(
        f"Total runtime: {runtime_s:.2f} seconds = "
        f"{runtime_s/SECONDS_PER_MINUTE:.2f} minutes"
    )
