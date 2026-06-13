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
    GEOMETRIC_FRACTION_HEATMAP_FILENAME,
    METADATA_FILENAME,
    NSC_RADIUS_REFERENCE_MASS_MSUN,
    RATE_HEATMAP_FILENAME,
    RATE_VS_SMBH_PLOT_FILENAME,
    TIMESCALE_HISTOGRAM_FILENAME,
    TIMESCALES_FILENAME,
)
from .config import TDEDynConfig
from .models import TIMESCALE_COLUMNS, TDEDynAveragedResult


def save_results(
    averaged_results: Sequence[TDEDynAveragedResult],
    output_dir: Path,
) -> pd.DataFrame:
    """Save averaged results with units encoded in column names."""
    frame = pd.DataFrame([result.to_dict() for result in averaged_results])
    path = output_dir / AVERAGED_RESULTS_FILENAME
    frame.to_csv(path, index=False, float_format="%.6e")
    print(f"saved: {path}")
    return frame


def save_timescales(
    averaged_results: Sequence[TDEDynAveragedResult],
    output_dir: Path,
) -> pd.DataFrame:
    """Save one row per damped primary or resupplied object."""
    records = [
        record
        for averaged_result in averaged_results
        for result in averaged_result.results
        for record in result.timescale_records
    ]
    frame = pd.DataFrame.from_records(
        records, columns=TIMESCALE_COLUMNS
    )
    path = output_dir / TIMESCALES_FILENAME
    frame.to_csv(path, index=False, float_format="%.6e")
    print(f"saved: {path}")
    return frame


def print_timescale_statistics(timescales_csv: Path) -> pd.DataFrame:
    """Print robust summary statistics read from ``timescales.csv``."""
    frame = pd.read_csv(timescales_csv)
    columns = {
        "damping_time_myr": "Damping",
        "migration_time_myr": "Migration",
        "relaxation_time_myr": "Relaxation",
        "delivery_time_myr": "Delivery",
    }
    rows: list[dict[str, float | int | str]] = []
    print("\nTimescale statistics [Myr]")
    print(
        f"  objects={len(frame)}, "
        f"primary={(frame['task_kind'] == 'primary').sum()}, "
        f"copies={(frame['task_kind'] == 'copy').sum()}, "
        f"counted TDEs={frame['counted_as_tde'].astype(bool).sum()}"
    )
    for column, label in columns.items():
        values = pd.to_numeric(frame[column], errors="coerce")
        values = values[np.isfinite(values) & (values >= 0.0)]
        if values.empty:
            print(f"  {label:<10}: no finite values")
            continue
        row: dict[str, float | int | str] = {
            "timescale": column,
            "count": int(values.size),
            "min_myr": float(values.min()),
            "p16_myr": float(values.quantile(0.16)),
            "median_myr": float(values.median()),
            "mean_myr": float(values.mean()),
            "p84_myr": float(values.quantile(0.84)),
            "p95_myr": float(values.quantile(0.95)),
            "max_myr": float(values.max()),
        }
        rows.append(row)
        print(
            f"  {label:<10}: n={values.size:<7d} "
            f"median={row['median_myr']:.3e}, "
            f"mean={row['mean_myr']:.3e}, "
            f"p16-p84={row['p16_myr']:.3e}-"
            f"{row['p84_myr']:.3e}"
        )
    return pd.DataFrame(rows)


def create_timescale_histograms_from_csv(
    timescales_csv: Path,
    output_dir: Path,
) -> None:
    """Create logarithmic histograms from persisted timescale data."""
    frame = pd.read_csv(timescales_csv)
    panels = (
        ("damping_time_myr", r"$t_{\rm damp}$"),
        ("migration_time_myr", r"$t_{\rm mig}$"),
        ("relaxation_time_myr", r"$t_{\rm relax}$"),
        ("delivery_time_myr", r"$t_{\rm damp}+t_{\rm mig}$"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    for axis, (column, label) in zip(axes.flat, panels):
        plotted = False
        all_values = pd.to_numeric(frame[column], errors="coerce")
        positive = all_values[np.isfinite(all_values) & (all_values > 0.0)]
        if not positive.empty:
            minimum = float(positive.min())
            maximum = float(positive.max())
            if np.isclose(minimum, maximum):
                bins = np.logspace(
                    np.log10(minimum) - 0.25,
                    np.log10(maximum) + 0.25,
                    12,
                )
            else:
                bins = np.logspace(
                    np.log10(minimum), np.log10(maximum), 31
                )
            for task_kind, color in (
                ("primary", "tab:blue"),
                ("copy", "tab:orange"),
            ):
                values = pd.to_numeric(
                    frame.loc[frame["task_kind"] == task_kind, column],
                    errors="coerce",
                )
                values = values[
                    np.isfinite(values) & (values > 0.0)
                ]
                if not values.empty:
                    axis.hist(
                        values,
                        bins=bins,
                        alpha=0.65,
                        label=task_kind,
                        color=color,
                    )
                    plotted = True
            axis.set_xscale("log")
        if not plotted:
            axis.text(
                0.5,
                0.5,
                "No positive finite values",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
        axis.set_xlabel(f"{label} [Myr]")
        axis.set_ylabel("Object count")
        axis.grid(True, alpha=0.25, linestyle="--")
        if plotted:
            axis.legend()
    figure.suptitle("TDE workflow timescale distributions")
    figure.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / TIMESCALE_HISTOGRAM_FILENAME
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"saved: {path}")


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
            "nsc_radius_reference_mass_msun": (NSC_RADIUS_REFERENCE_MASS_MSUN),
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
    """Create rate and geometric-selection plots."""
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
    geometry_grid = frame.pivot(
        index="m_smbh_msun",
        columns="m_nsc_msun",
        values="f_geo_mean",
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

    figure, axis = plt.subplots(figsize=(10, 8))
    image = axis.pcolormesh(
        smbh_masses_msun,
        nsc_masses_msun,
        geometry_grid.T,
        shading="auto",
        cmap="magma",
    )
    _configure_heatmap_axes(axis, smbh_masses_msun, nsc_masses_msun)
    axis.set_title(r"Geometric selection fraction $f_{\rm geo}$")
    figure.colorbar(image)
    figure.tight_layout()
    figure.savefig(
        output_dir / GEOMETRIC_FRACTION_HEATMAP_FILENAME,
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
