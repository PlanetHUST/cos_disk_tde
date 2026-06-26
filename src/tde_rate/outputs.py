"""Persistence and visualization for TDE-rate simulations."""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from constants import HUBBLE_TIME_MYR

from .config import (
    AVERAGED_RESULTS_FILENAME,
    COMBINED_TIMESCALE_HISTOGRAM_FILENAME,
    COPY_TIMESCALE_HISTOGRAM_FILENAME,
    GEOMETRIC_FRACTION_HEATMAP_FILENAME,
    METADATA_FILENAME,
    NSC_RADIUS_REFERENCE_MASS_MSUN,
    PRIMARY_TIMESCALE_HISTOGRAM_FILENAME,
    RATE_HEATMAP_FILENAME,
    RATE_VS_SMBH_PLOT_FILENAME,
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
    frame = pd.DataFrame.from_records(records, columns=TIMESCALE_COLUMNS)
    path = output_dir / TIMESCALES_FILENAME
    frame.to_csv(path, index=False, float_format="%.6e")
    print(f"saved: {path}")
    return frame


def print_timescale_statistics(timescales_csv: Path) -> pd.DataFrame:
    """Print descriptive timescale statistics read from ``timescales.csv``."""
    frame = pd.read_csv(timescales_csv)
    columns = {
        "damping_time_myr": "Damping",
        "migration_time_myr": "Migration",
        "relaxation_time_myr": "Relaxation",
        "delivery_time_myr": "Delivery",
    }
    counted_as_tde = _as_bool_series(frame["counted_as_tde"])
    rows: list[dict[str, float | int | str]] = []
    print("\nTimescale statistics [Myr]")
    print(
        f"  objects={len(frame)}, "
        f"primary={(frame['task_kind'] == 'primary').sum()}, "
        f"copies={(frame['task_kind'] == 'copy').sum()}, "
        f"counted TDEs={counted_as_tde.sum()}"
    )
    for column, label in columns.items():
        values = _valid_timescale_values(frame[column])
        if values.empty:
            print(f"  {label:<10}: no finite values")
            continue
        positive_values = values[values > 0.0]
        row: dict[str, float | int | str] = {
            "timescale": column,
            "count": int(values.size),
            "positive_count": int(positive_values.size),
            "min_myr": float(values.min()),
            "p05_myr": float(values.quantile(0.05)),
            "p16_myr": float(values.quantile(0.16)),
            "p25_myr": float(values.quantile(0.25)),
            "median_myr": float(values.median()),
            "mean_myr": float(values.mean()),
            "std_myr": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "p75_myr": float(values.quantile(0.75)),
            "p84_myr": float(values.quantile(0.84)),
            "p95_myr": float(values.quantile(0.95)),
            "max_myr": float(values.max()),
        }
        if positive_values.empty:
            row["log10_median_myr"] = np.nan
        else:
            row["log10_median_myr"] = float(
                np.log10(positive_values).median()
            )
        rows.append(row)
        print(
            f"  {label:<10}: n={values.size:<7d} "
            f"positive={positive_values.size:<7d} "
            f"median={row['median_myr']:.3e}, "
            f"mean={row['mean_myr']:.3e}, "
            f"std={row['std_myr']:.3e}, "
            f"IQR={row['p25_myr']:.3e}-"
            f"{row['p75_myr']:.3e}, "
            f"p16-p84={row['p16_myr']:.3e}-"
            f"{row['p84_myr']:.3e}, "
            f"p05-p95={row['p05_myr']:.3e}-"
            f"{row['p95_myr']:.3e}"
        )
    _print_grouped_timescale_statistics(frame, columns)
    _print_timescale_outcome_counts(frame)
    return pd.DataFrame(rows)


def _valid_timescale_values(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values[np.isfinite(values) & (values >= 0.0)]


def _as_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.lower().isin(("true", "1", "yes"))


def _format_group_timescale_summary(values: pd.Series) -> str:
    values = _valid_timescale_values(values)
    if values.empty:
        return "no finite values"
    return (
        f"n={values.size}, "
        f"median={values.median():.3e}, "
        f"IQR={values.quantile(0.25):.3e}-{values.quantile(0.75):.3e}, "
        f"p16-p84={values.quantile(0.16):.3e}-{values.quantile(0.84):.3e}"
    )


def _print_grouped_timescale_statistics(
    frame: pd.DataFrame,
    columns: dict[str, str],
) -> None:
    print("\nTimescale statistics by task kind [Myr]")
    for task_kind in ("primary", "copy"):
        selected = frame.loc[frame["task_kind"] == task_kind]
        print(f"  {task_kind}: objects={len(selected)}")
        if selected.empty:
            continue
        for column, label in columns.items():
            print(
                f"    {label:<10}: "
                f"{_format_group_timescale_summary(selected[column])}"
            )

    counted = frame.loc[_as_bool_series(frame["counted_as_tde"])]
    if counted.empty:
        print("\nCounted TDE delivery times [Myr]: no counted TDEs")
        return
    print("\nCounted TDE delivery times by mode [Myr]")
    group_columns = ["task_kind", "tde_mode"]
    for group_key, selected in counted.groupby(group_columns, sort=True):
        task_kind, tde_mode = group_key
        print(
            f"  {task_kind}/{tde_mode}: "
            f"{_format_group_timescale_summary(selected['delivery_time_myr'])}"
        )


def _print_timescale_outcome_counts(frame: pd.DataFrame) -> None:
    if "outcome" not in frame.columns:
        return
    print("\nTimescale outcome counts")
    counts = (
        frame.groupby(["task_kind", "outcome"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["task_kind", "count"], ascending=[True, False])
    )
    for row in counts.itertuples(index=False):
        print(f"  {row.task_kind}/{row.outcome}: {row.count}")


def create_timescale_histograms_from_csv(
    timescales_csv: Path,
    output_dir: Path,
) -> None:
    """Create primary, copied, and combined timescale PDFs."""
    frame = pd.read_csv(timescales_csv)
    series = (
        ("damping_time_myr", r"$t_{\rm damp}$", "-", "tab:blue"),
        ("migration_time_myr", r"$t_{\rm mig}$", "--", "tab:orange"),
        ("relaxation_time_myr", r"$t_{\rm relax}$", "-.", "tab:green"),
        (
            "delivery_time_myr",
            r"$t_{\rm tde}= t_{\rm damp}+t_{\rm mig}$",
            ":",
            "tab:red",
        ),
    )
    lower_log_time = -4.0
    upper_log_time = float(np.log10(HUBBLE_TIME_MYR))
    bins = np.linspace(lower_log_time, upper_log_time, 36)

    def create_figure(
        selected: pd.DataFrame,
        title: str,
        filename: str,
    ) -> None:
        figure, axis = plt.subplots(figsize=(10, 6))
        log_series: list[tuple[np.ndarray, str, str, str]] = []
        for column, label, linestyle, color in series:
            values = pd.to_numeric(selected[column], errors="coerce").to_numpy(
                dtype=float
            )
            values = values[
                np.isfinite(values)
                & (values >= 10.0**lower_log_time)
                & (values <= HUBBLE_TIME_MYR)
            ]
            if values.size:
                log_series.append((np.log10(values), label, linestyle, color))

        plotted = False
        if log_series:
            for values, label, linestyle, color in log_series:
                axis.hist(
                    values,
                    bins=bins,
                    density=True,
                    histtype="step",
                    linewidth=2.0,
                    linestyle=linestyle,
                    color=color,
                    label=f"{label}",
                )
                plotted = True

        axis.set_xlim(lower_log_time, upper_log_time)
        if not plotted:
            axis.text(
                0.5,
                0.5,
                "No positive finite values",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
        axis.set_xlabel(r"$\log_{10}(t/{\rm Myr})$", fontsize=16)
        axis.set_ylabel("Probability density function", fontsize=16)
        # axis.set_yscale("log")
        axis.tick_params(axis="both", which="major", labelsize=14)
        # axis.set_title(title, fontsize=20, pad=14)
        axis.grid(True, which="both", alpha=0.5, linestyle=":", linewidth=1.0)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        if plotted:
            _, upper_y = axis.get_ylim()
            axis.set_ylim(1e-4, upper_y * 1.05)
            axis.legend(
                loc="best",
                frameon=False,
                ncols=1,
                fontsize=16,
            )
        figure.tight_layout()
        path = output_dir / filename
        figure.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(figure)
        print(f"saved: {path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    create_figure(
        frame.loc[frame["task_kind"] == "primary"],
        "Timescale distributions: primary objects",
        PRIMARY_TIMESCALE_HISTOGRAM_FILENAME,
    )
    create_figure(
        frame.loc[frame["task_kind"] == "copy"],
        "Timescale distributions: copied objects",
        COPY_TIMESCALE_HISTOGRAM_FILENAME,
    )
    create_figure(
        frame.loc[frame["task_kind"].isin(("primary", "copy"))],
        "Timescale distributions: primary and copied objects",
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
