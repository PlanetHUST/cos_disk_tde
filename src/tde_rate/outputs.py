"""Persistence and visualization for TDE-rate simulations."""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from .config import (
    AVERAGED_RESULTS_FILENAME,
    COMBINED_TIMESCALE_HISTOGRAM_FILENAME,
    COPY_TIMESCALE_HISTOGRAM_FILENAME,
    METADATA_FILENAME,
    PRIMARY_TIMESCALE_HISTOGRAM_FILENAME,
    RATE_HEATMAP_FILENAME,
    RATE_VS_SMBH_PLOT_FILENAME,
    TIMESCALES_FILENAME,
)
from .config import TDEDynConfig
from .models import TDEDynAveragedResult


RESULT_COLUMNS = (
    "m_smbh_msun",
    "m_nsc_msun",
    "tde_mass_primary_mean_msun",
    "tde_mass_total_mean_msun",
    "tde_mass_rate_total_mean_msun_per_myr",
    "tde_mass_rate_total_min_msun_per_myr",
    "tde_mass_rate_total_max_msun_per_myr",
    "tde_star_count_total_mean",
)
TDE_EVENT_COLUMNS = (
    "m_smbh_msun",
    "m_nsc_msun",
    "repeat",
    "task_kind",
    "event_time_myr",
    "event_mass_msun",
    "cumulative_tde_mass_msun",
)


def save_results(
    averaged_results: Sequence[TDEDynAveragedResult],
    output_dir: Path,
) -> pd.DataFrame:
    """Save averaged results with units encoded in column names."""
    frame = pd.DataFrame(
        [result.to_dict() for result in averaged_results],
        columns=RESULT_COLUMNS,
    )
    path = output_dir / AVERAGED_RESULTS_FILENAME
    frame.to_csv(path, index=False, float_format="%.6e")
    print(f"saved: {path}")
    return frame


def save_timescales(
    averaged_results: Sequence[TDEDynAveragedResult],
    output_dir: Path,
) -> pd.DataFrame:
    """Save one row per TDE event, ordered by occurrence time."""
    records: list[dict[str, float | int | str]] = []
    for averaged_result in averaged_results:
        for result in averaged_result.results:
            events = [
                (event_time, event_mass, task_kind)
                for times, masses, task_kind in (
                    (
                        result.tde_times_primary_myr,
                        result.tde_masses_primary_msun,
                        "primary",
                    ),
                    (
                        result.tde_times_copies_myr,
                        result.tde_masses_copies_msun,
                        "copy",
                    ),
                )
                for event_time, event_mass in zip(times, masses, strict=True)
            ]
            cumulative_mass_msun = 0.0
            for event_time, event_mass, task_kind in sorted(events):
                cumulative_mass_msun += event_mass
                records.append(
                    {
                        "m_smbh_msun": averaged_result.m_smbh_msun,
                        "m_nsc_msun": averaged_result.m_nsc_msun,
                        "repeat": result.repeat_idx,
                        "task_kind": task_kind,
                        "event_time_myr": event_time,
                        "event_mass_msun": event_mass,
                        "cumulative_tde_mass_msun": cumulative_mass_msun,
                    }
                )
    frame = pd.DataFrame.from_records(records, columns=TDE_EVENT_COLUMNS)
    path = output_dir / TIMESCALES_FILENAME
    frame.to_csv(path, index=False, float_format="%.6e")
    print(f"saved: {path}")
    return frame


def print_timescale_statistics(timescales_csv: Path) -> pd.DataFrame:
    """Print TDE event statistics read from ``timescales.csv``."""
    frame = pd.read_csv(timescales_csv)
    print("\nTDE event statistics")
    rows = []
    for task_kind in ("primary", "copy", "total"):
        selected = (
            frame
            if task_kind == "total"
            else frame[frame["task_kind"] == task_kind]
        )
        row = {
            "task_kind": task_kind,
            "event_count": len(selected),
            "tde_mass_msun": selected["event_mass_msun"].sum(),
        }
        rows.append(row)
        print(
            f"  {task_kind}: events={row['event_count']}, "
            f"mass={row['tde_mass_msun']:.3e} Msun"
        )
    return pd.DataFrame(rows)


def create_timescale_histograms_from_csv(
    timescales_csv: Path,
    output_dir: Path,
) -> None:
    """Create cumulative disrupted-mass profiles for TDE events."""
    frame = pd.read_csv(timescales_csv)

    def create_figure(
        selected: pd.DataFrame,
        title: str,
        filename: str,
    ) -> None:
        figure, axis = plt.subplots(figsize=(10, 6))
        plotted = False
        group_columns = ["m_smbh_msun", "m_nsc_msun", "repeat"]
        for group_key, events in selected.groupby(group_columns, sort=True):
            events = events.sort_values("event_time_myr")
            axis.step(
                events["event_time_myr"],
                events["event_mass_msun"].cumsum(),
                where="post",
                alpha=0.45,
                linewidth=1.2,
                label=(
                    rf"$M_{{\rm BH}}={group_key[0]:.1e}$, "
                    rf"$M_{{\rm NSC}}={group_key[1]:.1e}$, r={group_key[2]}"
                )
            )
            plotted = True

        if not plotted:
            axis.text(
                0.5,
                0.5,
                "No positive finite values",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
        axis.set_xlabel(r"TDE event time $t\,[{\rm Myr}]$", fontsize=16)
        axis.set_ylabel(r"Cumulative TDE mass $[M_\odot]$", fontsize=16)
        axis.tick_params(axis="both", which="major", labelsize=14)
        axis.set_title(title, fontsize=18, pad=14)
        axis.grid(True, which="both", alpha=0.5, linestyle=":", linewidth=1.0)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        if plotted:
            axis.legend(
                loc="best",
                frameon=False,
                fontsize=8,
            )
        figure.tight_layout()
        path = output_dir / filename
        figure.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(figure)
        print(f"saved: {path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    create_figure(
        frame.loc[frame["task_kind"] == "primary"],
        "Cumulative TDE mass: primary stars",
        PRIMARY_TIMESCALE_HISTOGRAM_FILENAME,
    )
    create_figure(
        frame.loc[frame["task_kind"] == "copy"],
        "Cumulative TDE mass: copied stars",
        COPY_TIMESCALE_HISTOGRAM_FILENAME,
    )
    create_figure(
        frame.loc[frame["task_kind"].isin(("primary", "copy"))],
        "Cumulative TDE mass: primary and copied stars",
        COMBINED_TIMESCALE_HISTOGRAM_FILENAME,
    )


def save_metadata(
    output_dir: Path,
    config: TDEDynConfig,
    runtime_stats: dict[str, float | int],
) -> None:
    """Save simulation configuration and runtime statistics."""
    metadata: dict[str, Any] = {
        "simulation_info": {
            "date": datetime.now().isoformat(),
            "timestamp_s_since_epoch": time.time(),
            "description": (
                "Dynamical TDE mass-rate simulation with unified " "parallel damping"
            ),
        },
        "parameters": {
            "num_samples": config.num_samp,
            "num_repeats": config.num_repeats,
            "total_simulations": (config.num_samp**2 * config.num_repeats),
            "duty_cycle_myr": config.duty_cycle_myr,
            "m_smbh_range_msun": [
                config.m_smbh_min_msun,
                config.m_smbh_max_msun,
            ],
            "m_nsc_range_msun": [
                config.m_nsc_min_msun,
                config.m_nsc_max_msun,
            ],
            "metallicity": config.metallicity,
            "nsc_radius_pc_at_reference_mass": (config.nsc_radius_pc_at_reference_mass),
            "nsc_radius_reference_mass_msun": (
                config.nsc_radius_reference_mass_msun
            ),
            "nsc_radius_mass_slope": config.nsc_radius_mass_slope,
            "dyn_resupply": config.dyn_resupply,
            "max_copies_per_object": config.max_copies_per_object,
            "reuse_same_orbit_for_copies": (config.reuse_same_orbit_for_copies),
            "tde_radius_rg": config.tde_radius_rg,
            "direct_tde": config.direct_tde,
            "damping_cpu_count": config.damping_cpu_count,
            "checkpoint_grid_interval": (config.checkpoint_grid_interval),
            "create_visualizations": config.create_visualizations,
            "print_debug_results": config.print_debug_results,
        },
        "runtime_statistics": runtime_stats,
    }
    path = output_dir / METADATA_FILENAME
    with path.open("w") as metadata_file:
        json.dump(metadata, metadata_file, indent=2, default=str)
    print(f"saved: {path}")


def create_visualizations_from_csv(csv_file: Path, output_dir: Path) -> None:
    """Create result-oriented TDE-rate plots."""
    frame = pd.read_csv(csv_file)
    if frame.empty:
        print("No data to visualize")
        return

    smbh_masses_msun = np.sort(frame["m_smbh_msun"].unique())
    nsc_masses_msun = np.sort(frame["m_nsc_msun"].unique())
    rate_grid = frame.pivot(
        index="m_smbh_msun",
        columns="m_nsc_msun",
        values="tde_mass_rate_total_mean_msun_per_myr",
    ).values

    figure, axis = plt.subplots(figsize=(10, 8))
    for nsc_mass_msun in nsc_masses_msun:
        subset = frame[frame["m_nsc_msun"] == nsc_mass_msun].sort_values("m_smbh_msun")
        axis.plot(
            subset["m_smbh_msun"],
            subset["tde_mass_rate_total_mean_msun_per_myr"],
            "o-",
            label=f"M_NSC={nsc_mass_msun:.1e}",
        )
    axis.set_xscale("log")
    if np.any(frame["tde_mass_rate_total_mean_msun_per_myr"] > 0.0):
        axis.set_yscale("log")
    axis.set_xlabel(r"$M_{\rm SMBH}\,[M_\odot]$")
    axis.set_ylabel(r"TDE mass rate $[M_\odot/{\rm Myr}]$")
    axis.set_title("Dynamical TDE mass rate vs SMBH mass")
    axis.grid(True, alpha=0.3, linestyle="--")
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        output_dir / RATE_VS_SMBH_PLOT_FILENAME,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 8))
    image = axis.pcolormesh(
        smbh_masses_msun,
        nsc_masses_msun,
        rate_grid.T,
        shading="auto",
        norm="log" if np.any(rate_grid > 0.0) else None,
        cmap="viridis",
    )
    _configure_heatmap_axes(axis, smbh_masses_msun, nsc_masses_msun)
    axis.set_title(r"Dynamical TDE mass rate $[M_\odot/{\rm Myr}]$")
    figure.colorbar(image)
    figure.tight_layout()
    figure.savefig(
        output_dir / RATE_HEATMAP_FILENAME,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(figure)

    print(f"visualizations saved to: {output_dir}")


def _configure_heatmap_axes(
    axis: Any,
    smbh_masses_msun: np.ndarray,
    nsc_masses_msun: np.ndarray,
) -> None:
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel(r"$M_{\rm SMBH}\,[M_\odot]$")
    axis.set_ylabel(r"$M_{\rm NSC}\,[M_\odot]$")
    if smbh_masses_msun.size > 1:
        axis.set_xlim(smbh_masses_msun.min(), smbh_masses_msun.max())
    if nsc_masses_msun.size > 1:
        axis.set_ylim(nsc_masses_msun.min(), nsc_masses_msun.max())
