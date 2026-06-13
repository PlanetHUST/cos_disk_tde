"""
tde_dyn.py

Unified-parallel TDE mass-rate simulation for a dynamical NSC-AGN system.

Main ideas:
1. Sample the full NSC stellar population.
2. Select only stars whose orbits geometrically intersect the AGN disk.
3. Run primary damping in parallel using a unified damping-task framework.
4. Generate relaxation-driven copies from the primary results.
5. Run copy damping in parallel using the same framework.
6. Count TDEs from stars reaching the tidal-disruption region.

Outputs:
- tde_dyn_averaged_results.csv
- simulation_metadata.json

Required local modules:
    sampling as sstar
    helper_function as helpf
    general_class as agn
    damp import DampingCalculator, OrbitalState
    constants as const
    disk_updated as disk
"""

import os
import json
import sys
import time
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Optional, Sequence, TypedDict

import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib import pyplot as plt

import sampling as sstar
import helper_function as helpf
import general_class as agn
from damp import DampingCalculator, OrbitalState
import constants as const
import disk_updated as disk


# ============================================================
# Unit conventions and model constants
# ============================================================
# The simulation uses cgs internally. Configuration masses are in solar
# masses and configuration times are in Myr unless their names say otherwise.
YEARS_PER_MYR = 1.0e6
SECONDS_PER_MINUTE = 60.0
DISK_INNER_RADIUS_RG = 6.0
DISK_ACCRETION_RATE_G_S_PER_MSUN = 1.44e18
NSC_RADIUS_REFERENCE_MASS_MSUN = 3.6e6
NSC_RADIUS_LOG10_OFFSET = -0.011
MIN_RELAXATION_TIME_MYR = 1.0e-12
AVERAGED_RESULTS_FILENAME = "tde_dyn_averaged_results.csv"
METADATA_FILENAME = "simulation_metadata.json"
VISUALIZATION_DIRECTORY = "visualizations"
RATE_VS_SMBH_PLOT_FILENAME = "tde_dyn_rate_vs_smbh.png"
RATE_HEATMAP_FILENAME = "tde_dyn_rate_heatmap.png"
GEOMETRIC_FRACTION_HEATMAP_FILENAME = "tde_dyn_fgeo_heatmap.png"

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


class DampingTask(TypedDict):
    """One damping job; dimensional fields include their units in the key."""

    task_kind: str
    idx: int
    origin_primary_idx: int
    copy_number: int
    mass_g: float
    semi_major_axis_cm: float
    eccentricity: float
    inclination_rad: float
    cos_omega: float
    formation_time_myr: float
    state_time_s: float
    type_flag: int


class DampingResult(TypedDict):
    """Result from one damping job, with explicit dimensional field names."""

    task_kind: str
    idx: int
    origin_primary_idx: int
    copy_number: int
    final_time_s: float
    final_time_myr: float
    final_semi_major_axis_cm: float
    obj_type: int
    formation_time_myr: float
    initial_semi_major_axis_cm: float
    eccentricity: float
    inclination_rad: float
    cos_omega: float
    mass_g: float
    mass_msun: float


class DiskProfiles(TypedDict):
    """AGN disk profiles in cgs units."""

    inner_radius_cm: float
    outer_radius_cm: float
    radius_cm: FloatArray
    surface_density_g_cm2: FloatArray
    aspect_ratio: FloatArray
    gravitational_radius_cm: float


class StellarPopulation(TypedDict):
    """Sampled NSC population and orbital elements."""

    mass_msun: FloatArray
    formation_time_myr: FloatArray
    semi_major_axis_cm: FloatArray
    eccentricity: FloatArray
    inclination_rad: FloatArray
    cos_omega: FloatArray
    nsc_radius_cm: float


class GeometricPopulation(TypedDict):
    """Subset of stellar orbits that geometrically cross the disk."""

    mass_msun: FloatArray
    formation_time_myr: FloatArray
    semi_major_axis_cm: FloatArray
    eccentricity: FloatArray
    inclination_rad: FloatArray
    cos_omega: FloatArray


# ============================================================
# Verbose helper
# ============================================================
VERBOSE = True


def vprint(*args: Any, **kwargs: Any) -> None:
    if VERBOSE:
        print(*args, **kwargs)


# ============================================================
# HPC-aware CPU count
# ============================================================
def get_hpc_cpu_count() -> int:
    """
    Prefer scheduler-assigned CPU counts when available.
    Falls back to helper_function.get_cpu_count().
    """
    for var in ("SLURM_CPUS_PER_TASK", "PBS_NP", "NSLOTS"):
        val = os.environ.get(var)
        if val is not None:
            try:
                return max(1, int(val))
            except ValueError:
                pass
    return helpf.get_cpu_count()


# ============================================================
# Unified damping worker globals
# ============================================================
_WORKER_CALCULATOR: Optional[DampingCalculator] = None


def init_damping_worker(
    agn_properties: agn.AGN_properties,
    radius_cm: FloatArray,
    surface_density_g_cm2: FloatArray,
    aspect_ratio: FloatArray,
) -> None:
    """
    Initialize one DampingCalculator per worker process.
    This avoids rebuilding the calculator for every task.
    """
    global _WORKER_CALCULATOR
    _WORKER_CALCULATOR = DampingCalculator(
        agn_properties,
        verbose=False,
        r_list=radius_cm,
        sigma_list=surface_density_g_cm2,
        h_list=aspect_ratio,
    )


def process_damping_task(task: DampingTask) -> DampingResult:
    """
    Unified worker for both primary and copy damping tasks.

    Task format:
        {
            "task_kind": "primary" or "copy",
            "idx": int,
            "origin_primary_idx": int,
            "copy_number": int,
            "mass_g": float,
            "semi_major_axis_cm": float,
            "eccentricity": float,
            "inclination_rad": float,
            "cos_omega": float,
            "formation_time_myr": float,
            "state_time_s": float,
            "type_flag": int,
        }
    """
    global _WORKER_CALCULATOR
    if _WORKER_CALCULATOR is None:
        raise RuntimeError("Damping worker was not initialized")

    try:
        state = OrbitalState(
            task["semi_major_axis_cm"],
            task["eccentricity"],
            task["inclination_rad"],
            task["cos_omega"],
            task["state_time_s"],
        )

        final_time_s, final_semi_major_axis_cm, obj_type = (
            _WORKER_CALCULATOR.damping_simulation(
                task["mass_g"],
                state,
                task["formation_time_myr"],
                "Star",
                task["type_flag"],
            )
        )

        return {
            "task_kind": task["task_kind"],
            "idx": task["idx"],
            "origin_primary_idx": task["origin_primary_idx"],
            "copy_number": task["copy_number"],
            "final_time_s": final_time_s,
            "final_time_myr": final_time_s / const.Myr,
            "final_semi_major_axis_cm": final_semi_major_axis_cm,
            "obj_type": obj_type,
            "formation_time_myr": task["formation_time_myr"],
            "initial_semi_major_axis_cm": task["semi_major_axis_cm"],
            "eccentricity": task["eccentricity"],
            "inclination_rad": task["inclination_rad"],
            "cos_omega": task["cos_omega"],
            "mass_g": task["mass_g"],
            "mass_msun": task["mass_g"] / const.msun,
        }

    except Exception:
        return {
            "task_kind": task["task_kind"],
            "idx": task["idx"],
            "origin_primary_idx": task["origin_primary_idx"],
            "copy_number": task["copy_number"],
            "final_time_s": 0.0,
            "final_time_myr": 0.0,
            "final_semi_major_axis_cm": 0.0,
            "obj_type": -1,
            "formation_time_myr": task["formation_time_myr"],
            "initial_semi_major_axis_cm": task["semi_major_axis_cm"],
            "eccentricity": task["eccentricity"],
            "inclination_rad": task["inclination_rad"],
            "cos_omega": task["cos_omega"],
            "mass_g": task["mass_g"],
            "mass_msun": task["mass_g"] / const.msun,
        }


def run_damping_tasks_parallel(
    tasks: Sequence[DampingTask],
    agn_properties: agn.AGN_properties,
    radius_cm: FloatArray,
    surface_density_g_cm2: FloatArray,
    aspect_ratio: FloatArray,
    cpu_count: int,
) -> list[DampingResult]:
    """
    Run a list of unified damping tasks in parallel.
    """
    if not tasks:
        return []

    num_processes = max(1, min(cpu_count, len(tasks)))
    chunksize = max(1, min(8, len(tasks) // max(1, 4 * num_processes)))

    vprint(
        f"Running {len(tasks)} damping tasks with cpus={num_processes}, chunksize={chunksize}"
    )

    with Pool(
        processes=num_processes,
        initializer=init_damping_worker,
        initargs=(
            agn_properties,
            radius_cm,
            surface_density_g_cm2,
            aspect_ratio,
        ),
    ) as pool:
        results = list(
            pool.imap_unordered(process_damping_task, tasks, chunksize=chunksize)
        )

    return results


# ============================================================
# Configuration
# ============================================================
class TDEDynConfig:
    """Configuration for the dynamical TDE simulation.

    Mass inputs are in solar masses, times are in Myr, and radii ending in
    ``_pc`` or ``_rg`` are measured in parsecs or gravitational radii.
    """

    def __init__(
        self,
        num_samp: int = 3,
        num_repeats: int = 3,
        duty_cycle_myr: float = 10.0,
        m_smbh_min_msun: float = 1e7,
        m_smbh_max_msun: float = 1e8,
        m_nsc_min_msun: float = 1e6,
        m_nsc_max_msun: float = 1e7,
        output_dir: str | Path = "./tde_dyn_results",
        metallicity: float = 0.01,
        nsc_radius_pc_at_reference_mass: float = 3.31,
        nsc_radius_mass_slope: float = 0.321,
        use_disk_inner_radius: bool = True,
        dyn_resupply: bool = True,
        max_copies_per_object: int = 30,
        reuse_same_orbit_for_copies: bool = True,
        tde_radius_rg: float = 10.0,
        direct_tde: bool = True,
        damping_cpu_count: Optional[int] = None,
    ) -> None:
        self.num_samp = num_samp
        self.num_repeats = num_repeats
        self.duty_cycle_myr = duty_cycle_myr
        self.m_smbh_min_msun = m_smbh_min_msun
        self.m_smbh_max_msun = m_smbh_max_msun
        self.m_nsc_min_msun = m_nsc_min_msun
        self.m_nsc_max_msun = m_nsc_max_msun
        self.output_dir = Path(output_dir)
        self.metallicity = metallicity

        # NSC size model:
        # R_NSC[pc] = R_ref[pc] * (M_NSC/M_ref)^alpha * 10^offset.
        self.nsc_radius_pc_at_reference_mass = nsc_radius_pc_at_reference_mass
        self.nsc_radius_mass_slope = nsc_radius_mass_slope

        self.use_disk_inner_radius = use_disk_inner_radius
        self.dyn_resupply = dyn_resupply
        self.max_copies_per_object = max_copies_per_object
        self.reuse_same_orbit_for_copies = reuse_same_orbit_for_copies
        self.tde_radius_rg = tde_radius_rg
        self.direct_tde = direct_tde
        self.damping_cpu_count = damping_cpu_count

    def get_param_grid(self) -> tuple[FloatArray, FloatArray]:
        """Return logarithmic SMBH and NSC mass grids in solar masses."""
        m_smbh_samples_msun = helpf.initial_mass(
            self.num_samp, self.m_smbh_min_msun, self.m_smbh_max_msun
        )
        m_nsc_samples_msun = helpf.initial_mass(
            self.num_samp, self.m_nsc_min_msun, self.m_nsc_max_msun
        )
        return (
            np.asarray(m_smbh_samples_msun, dtype=float),
            np.asarray(m_nsc_samples_msun, dtype=float),
        )


# ============================================================
# Result containers
# ============================================================
class TDEDynSingleResult:
    """Results for one repeat at one grid point."""

    def __init__(
        self, m_smbh_msun: float, m_nsc_msun: float, repeat_idx: int
    ) -> None:
        self.m_smbh_msun = m_smbh_msun
        self.m_nsc_msun = m_nsc_msun
        self.repeat_idx = repeat_idx

        # geometry
        self.total_stars = 0
        self.geo_selected_stars = 0
        self.f_geo = 0.0

        # primary damping results
        self.embedded_stars_primary = 0
        self.embedded_bh_primary = 0
        self.outside_primary = 0

        # dynamical copies
        self.num_copies_generated = 0
        self.num_copy_tdes = 0
        self.copy_tde_mass_msun = 0.0

        # TDE accounting
        self.tde_star_count_primary = 0
        self.tde_mass_primary_msun = 0.0

        self.tde_star_count_total = 0
        self.tde_mass_total_msun = 0.0
        self.tde_mass_rate_total_msun_per_myr = 0.0

        self.tde_masses_primary_msun: list[float] = []
        self.tde_masses_copies_msun: list[float] = []
        self.tde_times_primary_myr: list[float] = []
        self.tde_times_copies_myr: list[float] = []

        # diagnostics
        self.disk_inner_radius_cm: Optional[float] = None
        self.disk_outer_radius_cm: Optional[float] = None
        self.nsc_radius_cm: Optional[float] = None
        self.mean_relaxation_time_myr_geo = np.nan

        # debug counters
        self.direct_tde_primary_count = 0
        self.migration_tde_primary_count = 0
        self.direct_tde_copy_count = 0
        self.migration_tde_copy_count = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "m_smbh_msun": self.m_smbh_msun,
            "m_nsc_msun": self.m_nsc_msun,
            "repeat": self.repeat_idx,
            "total_stars": self.total_stars,
            "geo_selected_stars": self.geo_selected_stars,
            "f_geo": self.f_geo,
            "embedded_stars_primary": self.embedded_stars_primary,
            "embedded_bh_primary": self.embedded_bh_primary,
            "outside_primary": self.outside_primary,
            "num_copies_generated": self.num_copies_generated,
            "num_copy_tdes": self.num_copy_tdes,
            "tde_star_count_primary": self.tde_star_count_primary,
            "tde_mass_primary_msun": self.tde_mass_primary_msun,
            "copy_tde_mass_msun": self.copy_tde_mass_msun,
            "tde_star_count_total": self.tde_star_count_total,
            "tde_mass_total_msun": self.tde_mass_total_msun,
            "tde_mass_rate_total_msun_per_myr": (
                self.tde_mass_rate_total_msun_per_myr
            ),
            "disk_inner_radius_cm": self.disk_inner_radius_cm,
            "disk_outer_radius_cm": self.disk_outer_radius_cm,
            "nsc_radius_cm": self.nsc_radius_cm,
            "mean_relaxation_time_myr_geo": self.mean_relaxation_time_myr_geo,
            "tde_masses_primary_msun": self.tde_masses_primary_msun,
            "tde_masses_copies_msun": self.tde_masses_copies_msun,
            "tde_times_primary_myr": self.tde_times_primary_myr,
            "tde_times_copies_myr": self.tde_times_copies_myr,
        }


class TDEDynAveragedResult:
    """Average across repeats at one grid point."""

    def __init__(
        self,
        m_smbh_msun: float,
        m_nsc_msun: float,
        results: Sequence[TDEDynSingleResult],
    ) -> None:
        self.m_smbh_msun = m_smbh_msun
        self.m_nsc_msun = m_nsc_msun
        self.results = results

    def to_dict(self) -> dict[str, float | int]:
        def arr(name: str) -> FloatArray:
            return np.array(
                [getattr(r, name) for r in self.results], dtype=float
            )

        return {
            "m_smbh_msun": self.m_smbh_msun,
            "m_nsc_msun": self.m_nsc_msun,
            "num_repeats": len(self.results),
            "f_geo_mean": np.mean(arr("f_geo")),
            "f_geo_min": np.min(arr("f_geo")),
            "f_geo_max": np.max(arr("f_geo")),
            "geo_selected_stars_mean": np.mean(arr("geo_selected_stars")),
            "tde_mass_primary_mean_msun": np.mean(
                arr("tde_mass_primary_msun")
            ),
            "tde_mass_total_mean_msun": np.mean(arr("tde_mass_total_msun")),
            "tde_mass_rate_total_mean_msun_per_myr": np.mean(
                arr("tde_mass_rate_total_msun_per_myr")
            ),
            "tde_mass_rate_total_min_msun_per_myr": np.min(
                arr("tde_mass_rate_total_msun_per_myr")
            ),
            "tde_mass_rate_total_max_msun_per_myr": np.max(
                arr("tde_mass_rate_total_msun_per_myr")
            ),
            "tde_star_count_total_mean": np.mean(arr("tde_star_count_total")),
            "num_copies_generated_mean": np.mean(arr("num_copies_generated")),
            "mean_relaxation_time_geo_mean_myr": np.mean(
                arr("mean_relaxation_time_myr_geo")
            ),
        }


# ============================================================
# Main engine
# ============================================================
class TDEDynEngine:
    """Dynamical TDE engine with geometric selection + relaxation resupply."""

    def __init__(self, config: TDEDynConfig) -> None:
        self.config = config
        self.duty_cycle_myr = config.duty_cycle_myr
        self.disk_cache: dict[float, DiskProfiles] = {}

    # ----------------------------
    # NSC size model
    # ----------------------------
    def estimate_nsc_radius_cm(self, m_nsc_msun: float) -> float:
        """Estimate the NSC radius in cm.

        Uses
        ``R_NSC = R_ref (M_NSC / 3.6e6 Msun)^alpha 10^-0.011``,
        where ``R_ref`` is in pc and ``M_NSC`` is in solar masses.
        """
        nsc_radius_pc = (
            self.config.nsc_radius_pc_at_reference_mass
            * (m_nsc_msun / NSC_RADIUS_REFERENCE_MASS_MSUN)
            ** self.config.nsc_radius_mass_slope
        ) * 10**NSC_RADIUS_LOG10_OFFSET
        return float(nsc_radius_pc * const.parsec)

    # ----------------------------
    # Disk model cache
    # ----------------------------
    def get_disk_profiles(self, m_smbh_msun: float) -> DiskProfiles:
        """Return cached disk profiles for an SMBH mass in solar masses."""
        if m_smbh_msun not in self.disk_cache:
            accretion_rate_g_s = (
                DISK_ACCRETION_RATE_G_S_PER_MSUN * m_smbh_msun
            )
            (
                outer_radius_cm,
                radius_cm,
                _temperature_k,
                _toomre_q,
                surface_density_g_cm2,
                _optical_depth,
                aspect_ratio,
                _sound_speed_cm_s,
                _density_g_cm3,
            ) = disk.radial_structure(m_smbh_msun, accretion_rate_g_s)

            gravitational_radius_cm = (
                const.G * m_smbh_msun * const.msun / const.c**2
            )
            inner_radius_cm = (
                DISK_INNER_RADIUS_RG * gravitational_radius_cm
            )

            self.disk_cache[m_smbh_msun] = {
                "inner_radius_cm": float(inner_radius_cm),
                "outer_radius_cm": float(outer_radius_cm),
                "radius_cm": np.asarray(radius_cm, dtype=float),
                "surface_density_g_cm2": np.asarray(
                    surface_density_g_cm2, dtype=float
                ),
                "aspect_ratio": np.asarray(aspect_ratio, dtype=float),
                "gravitational_radius_cm": float(gravitational_radius_cm),
            }
            vprint(f"  cached disk for M_SMBH={m_smbh_msun:.3e} Msun")

        return self.disk_cache[m_smbh_msun]

    # ----------------------------
    # Geometry
    # ----------------------------
    def select_disk_crossing_stars(
        self,
        semi_major_axis_cm: FloatArray,
        eccentricity: FloatArray,
        cos_omega: FloatArray,
        disk_inner_radius_cm: float,
        disk_outer_radius_cm: float,
    ) -> tuple[BoolArray, FloatArray, FloatArray]:
        """Select disk-crossing orbits and return node radii in cm."""
        inner_node_radius_cm = (
            semi_major_axis_cm
            * (1.0 - eccentricity**2)
            / (1 + eccentricity * cos_omega)
        )
        outer_node_radius_cm = (
            semi_major_axis_cm
            * (1.0 - eccentricity**2)
            / (1 - eccentricity * cos_omega)
        )

        if self.config.use_disk_inner_radius:
            mask = (inner_node_radius_cm < disk_outer_radius_cm) & (
                outer_node_radius_cm > disk_inner_radius_cm
            )
        else:
            mask = inner_node_radius_cm < disk_outer_radius_cm

        return mask, inner_node_radius_cm, outer_node_radius_cm

    # ----------------------------
    # Full NSC sampling
    # ----------------------------
    def sample_full_population(
        self, m_smbh_msun: float, m_nsc_msun: float
    ) -> StellarPopulation:
        """Sample stellar masses, lifetimes, and NSC orbital elements.

        Stellar masses are returned in solar masses, formation times in Myr,
        orbital radii in cm, and inclinations in radians.
        """
        stellar_mass_msun, stellar_lifetime_yr = sstar.sample_efficient(
            m_nsc_msun, Z=self.config.metallicity
        )
        formation_time_myr = stellar_lifetime_yr / YEARS_PER_MYR

        num_stars = len(stellar_mass_msun)
        nsc_radius_cm = self.estimate_nsc_radius_cm(m_nsc_msun)

        (
            semi_major_axis_cm,
            eccentricity,
            inclination_rad,
            cos_omega,
        ) = helpf.initial_star(
            num_stars, m_smbh_msun, rmax=nsc_radius_cm
        )

        return {
            "mass_msun": np.asarray(stellar_mass_msun, dtype=float),
            "formation_time_myr": np.asarray(formation_time_myr, dtype=float),
            "semi_major_axis_cm": np.asarray(semi_major_axis_cm, dtype=float),
            "eccentricity": np.asarray(eccentricity, dtype=float),
            "inclination_rad": np.asarray(inclination_rad, dtype=float),
            "cos_omega": np.asarray(cos_omega, dtype=float),
            "nsc_radius_cm": nsc_radius_cm,
        }

    # ----------------------------
    # Unified task builders
    # ----------------------------
    def build_primary_tasks(
        self, geo_pool: GeometricPopulation, dpflags: agn.damp_flag
    ) -> list[DampingTask]:
        """Build primary stellar damping tasks using cgs orbital quantities."""
        tasks: list[DampingTask] = []
        for idx in range(len(geo_pool["mass_msun"])):
            tasks.append(
                {
                    "task_kind": "primary",
                    "idx": idx,
                    "origin_primary_idx": -1,
                    "copy_number": -1,
                    "mass_g": float(geo_pool["mass_msun"][idx] * const.msun),
                    "semi_major_axis_cm": float(
                        geo_pool["semi_major_axis_cm"][idx]
                    ),
                    "eccentricity": float(geo_pool["eccentricity"][idx]),
                    "inclination_rad": float(
                        geo_pool["inclination_rad"][idx]
                    ),
                    "cos_omega": float(geo_pool["cos_omega"][idx]),
                    "formation_time_myr": float(
                        geo_pool["formation_time_myr"][idx]
                    ),
                    "state_time_s": 0.0,
                    "type_flag": int(dpflags.type_flag[idx]),
                }
            )
        return tasks

    def build_copy_tasks(
        self,
        agn_properties: agn.AGN_properties,
        primary_results: Sequence[DampingResult],
        geo_pool: GeometricPopulation,
        relaxation_time_myr_geo: FloatArray,
    ) -> list[DampingTask]:
        """Build relaxation-resupply tasks with times in Myr and seconds."""
        if not self.config.dyn_resupply:
            return []

        tasks: list[DampingTask] = []
        geo_size = len(geo_pool["mass_msun"])
        if geo_size == 0:
            return tasks

        tde_radius_cm = self.config.tde_radius_rg * agn_properties.rg
        agn_lifetime_myr = agn_properties.AGNlifetime

        for res in primary_results:
            if res["obj_type"] not in (0, 1):
                continue
            if res["final_semi_major_axis_cm"] <= tde_radius_cm:
                continue

            relaxation_time_myr = max(
                float(relaxation_time_myr_geo[res["idx"]]),
                MIN_RELAXATION_TIME_MYR,
            )
            time_remaining_myr = max(
                0.0, agn_lifetime_myr - res["final_time_myr"]
            )

            expected_copies = time_remaining_myr / relaxation_time_myr
            num_copies = int(np.ceil(expected_copies))
            num_copies = min(num_copies, self.config.max_copies_per_object)

            if num_copies <= 0:
                continue

            for copy_number in range(num_copies):
                sampled_idx = np.random.randint(0, geo_size)

                sampled_mass_msun = geo_pool["mass_msun"][sampled_idx]
                sampled_formation_time_myr = geo_pool[
                    "formation_time_myr"
                ][sampled_idx]

                if self.config.reuse_same_orbit_for_copies:
                    initial_semi_major_axis_cm = res[
                        "initial_semi_major_axis_cm"
                    ]
                    eccentricity = res["eccentricity"]
                    inclination_rad = res["inclination_rad"]
                    cos_omega = res["cos_omega"]
                else:
                    initial_semi_major_axis_cm = float(
                        geo_pool["semi_major_axis_cm"][sampled_idx]
                    )
                    eccentricity = float(
                        geo_pool["eccentricity"][sampled_idx]
                    )
                    inclination_rad = float(
                        geo_pool["inclination_rad"][sampled_idx]
                    )
                    cos_omega = float(geo_pool["cos_omega"][sampled_idx])

                time_offset_myr = copy_number * relaxation_time_myr

                tasks.append(
                    {
                        "task_kind": "copy",
                        "idx": sampled_idx,
                        "origin_primary_idx": res["idx"],
                        "copy_number": copy_number,
                        "mass_g": float(sampled_mass_msun * const.msun),
                        "semi_major_axis_cm": initial_semi_major_axis_cm,
                        "eccentricity": eccentricity,
                        "inclination_rad": inclination_rad,
                        "cos_omega": cos_omega,
                        "formation_time_myr": float(
                            sampled_formation_time_myr + time_offset_myr
                        ),
                        "state_time_s": time_offset_myr * const.Myr,
                        "type_flag": 0,
                    }
                )

        return tasks

    # ----------------------------
    # Classification
    # ----------------------------
    def classify_primary_and_copies(
        self,
        result: TDEDynSingleResult,
        agn_properties: agn.AGN_properties,
        primary_results: Sequence[DampingResult],
        copies: Sequence[DampingResult],
        radius_cm: FloatArray,
        surface_density_g_cm2: FloatArray,
        aspect_ratio: FloatArray,
    ) -> None:
        """Classify direct and migration TDEs and accumulate mass in Msun."""
        tde_radius_cm = self.config.tde_radius_rg * agn_properties.rg

        embedded_stars_primary = 0
        embedded_stars_copies = 0
        embedded_bh = 0
        outside = 0

        # primary
        for res in primary_results:
            if res["obj_type"] == 0:
                if res["final_semi_major_axis_cm"] < tde_radius_cm:
                    if self.config.direct_tde:
                        result.direct_tde_primary_count += 1
                        result.tde_star_count_primary += 1
                        mass_msun = res["mass_msun"]
                        result.tde_mass_primary_msun += mass_msun
                        result.tde_masses_primary_msun.append(mass_msun)
                        result.tde_times_primary_myr.append(
                            res["final_time_myr"]
                        )
                else:
                    embedded_stars_primary += 1
                    migration_time_myr = helpf.t_migration(
                        agn_properties,
                        res["mass_g"],
                        res["initial_semi_major_axis_cm"],
                        h_list=aspect_ratio,
                        sigma_list=surface_density_g_cm2,
                        r_list=radius_cm,
                    )

                    if (
                        migration_time_myr + res["final_time_myr"]
                        < agn_properties.AGNlifetime
                    ):
                        result.migration_tde_primary_count += 1
                        result.tde_star_count_primary += 1
                        mass_msun = res["mass_msun"]
                        result.tde_mass_primary_msun += mass_msun
                        result.tde_masses_primary_msun.append(mass_msun)
                        result.tde_times_primary_myr.append(
                            float(
                                res["final_time_myr"] + migration_time_myr
                            )
                        )

            elif res["obj_type"] == 1:
                if res["final_semi_major_axis_cm"] >= tde_radius_cm:
                    embedded_bh += 1
            else:
                outside += 1

        result.embedded_stars_primary = embedded_stars_primary
        result.embedded_bh_primary = embedded_bh
        result.outside_primary = outside

        # copies
        result.num_copies_generated = len(copies)

        for res in copies:
            if res["obj_type"] == 0:
                if res["final_semi_major_axis_cm"] < tde_radius_cm:
                    if self.config.direct_tde:
                        result.direct_tde_copy_count += 1
                        result.num_copy_tdes += 1
                        result.copy_tde_mass_msun += res["mass_msun"]
                        result.tde_masses_copies_msun.append(res["mass_msun"])
                        result.tde_times_copies_myr.append(
                            res["final_time_myr"]
                        )
                else:
                    embedded_stars_copies += 1
                    migration_time_myr = helpf.t_migration(
                        agn_properties,
                        res["mass_g"],
                        res["initial_semi_major_axis_cm"],
                        r_list=radius_cm,
                        sigma_list=surface_density_g_cm2,
                        h_list=aspect_ratio,
                    )

                    if (
                        migration_time_myr + res["final_time_myr"]
                        < agn_properties.AGNlifetime
                    ):
                        result.migration_tde_copy_count += 1
                        result.num_copy_tdes += 1
                        result.copy_tde_mass_msun += res["mass_msun"]
                        result.tde_masses_copies_msun.append(res["mass_msun"])
                        result.tde_times_copies_myr.append(
                            float(
                                res["final_time_myr"] + migration_time_myr
                            )
                        )

        result.tde_star_count_total = (
            result.tde_star_count_primary + result.num_copy_tdes
        )
        result.tde_mass_total_msun = (
            result.tde_mass_primary_msun + result.copy_tde_mass_msun
        )
        result.tde_mass_rate_total_msun_per_myr = (
            result.tde_mass_total_msun / self.duty_cycle_myr
        )

        vprint(f" embedded stars primary: {embedded_stars_primary}")
        vprint(f" embedded stars copies: {embedded_stars_copies}")
        vprint(
            f" tde masses primary: {result.tde_mass_primary_msun:.3e} Msun"
        )
        vprint(f" tde masses copies: {result.copy_tde_mass_msun:.3e} Msun")

    # ----------------------------
    # One full repeat
    # ----------------------------
    def run_single_simulation(
        self, m_smbh_msun: float, m_nsc_msun: float, repeat_idx: int
    ) -> TDEDynSingleResult:
        """Run one Monte Carlo realization for masses supplied in Msun."""
        result = TDEDynSingleResult(m_smbh_msun, m_nsc_msun, repeat_idx)

        cpu_count = (
            self.config.damping_cpu_count
            if self.config.damping_cpu_count is not None
            else get_hpc_cpu_count()
        )

        # AGN
        agn_properties = agn.AGN_properties()
        agn_properties.init_AGN(m_smbh_msun, self.duty_cycle_myr)
        vprint("AGN initialized done.")

        # disk
        disk_data = self.get_disk_profiles(m_smbh_msun)
        disk_inner_radius_cm = disk_data["inner_radius_cm"]
        disk_outer_radius_cm = disk_data["outer_radius_cm"]
        radius_cm = disk_data["radius_cm"]
        surface_density_g_cm2 = disk_data["surface_density_g_cm2"]
        aspect_ratio = disk_data["aspect_ratio"]

        result.disk_inner_radius_cm = disk_inner_radius_cm
        result.disk_outer_radius_cm = disk_outer_radius_cm
        vprint(
            "Disk profiles cached. "
            f"r_out = {disk_outer_radius_cm:.3e} cm = "
            f"{disk_outer_radius_cm/const.parsec:.3e} pc"
        )

        # full NSC
        population = self.sample_full_population(m_smbh_msun, m_nsc_msun)
        result.nsc_radius_cm = population["nsc_radius_cm"]

        stellar_mass_msun = population["mass_msun"]
        formation_time_myr = population["formation_time_myr"]
        semi_major_axis_cm = population["semi_major_axis_cm"]
        eccentricity = population["eccentricity"]
        inclination_rad = population["inclination_rad"]
        cos_omega = population["cos_omega"]

        result.total_stars = len(stellar_mass_msun)

        # geometry
        mask_geo, _inner_node_radius_cm, _outer_node_radius_cm = (
            self.select_disk_crossing_stars(
                semi_major_axis_cm,
                eccentricity,
                cos_omega,
                disk_inner_radius_cm,
                disk_outer_radius_cm,
            )
        )
        geo_idx = np.where(mask_geo)[0]

        result.geo_selected_stars = len(geo_idx)
        result.f_geo = (
            len(geo_idx) / len(stellar_mass_msun)
            if len(stellar_mass_msun) > 0
            else 0.0
        )

        if len(geo_idx) == 0:
            return result

        vprint(
            f"Geometry selection: {len(geo_idx)}/{len(stellar_mass_msun)} "
            f"stars selected (f_geo={result.f_geo:.3e})"
        )

        # selected pool
        geo_pool: GeometricPopulation = {
            "mass_msun": stellar_mass_msun[geo_idx],
            "formation_time_myr": formation_time_myr[geo_idx],
            "semi_major_axis_cm": semi_major_axis_cm[geo_idx],
            "eccentricity": eccentricity[geo_idx],
            "inclination_rad": inclination_rad[geo_idx],
            "cos_omega": cos_omega[geo_idx],
        }

        # local relaxation times on selected orbits
        relaxation_time_s_geo = np.array(
            [
                helpf.t_relax(agn_properties.M_SMBH, orbit_radius_cm)
                for orbit_radius_cm in geo_pool["semi_major_axis_cm"]
            ],
            dtype=float,
        )
        relaxation_time_myr_geo = relaxation_time_s_geo / const.Myr
        result.mean_relaxation_time_myr_geo = (
            np.mean(relaxation_time_myr_geo)
            if len(relaxation_time_myr_geo) > 0
            else np.nan
        )
        vprint(
            "Relaxation times initialized, mean_trel_myr_geo = "
            f"{result.mean_relaxation_time_myr_geo:.3e} Myr"
        )

        # primary tasks
        dpflags = agn.damp_flag(len(geo_idx))
        primary_tasks = self.build_primary_tasks(geo_pool, dpflags)
        primary_results = run_damping_tasks_parallel(
            primary_tasks,
            agn_properties,
            radius_cm,
            surface_density_g_cm2,
            aspect_ratio,
            cpu_count=cpu_count,
        )
        primary_results = sorted(primary_results, key=lambda x: x["idx"])

        # copy tasks
        copy_tasks = self.build_copy_tasks(
            agn_properties,
            primary_results,
            geo_pool,
            relaxation_time_myr_geo,
        )
        copy_results = run_damping_tasks_parallel(
            copy_tasks,
            agn_properties,
            radius_cm,
            surface_density_g_cm2,
            aspect_ratio,
            cpu_count=cpu_count,
        )
        copy_results = sorted(
            copy_results,
            key=lambda x: (x["origin_primary_idx"], x["copy_number"]),
        )

        # classify
        self.classify_primary_and_copies(
            result,
            agn_properties,
            primary_results,
            copy_results,
            radius_cm,
            surface_density_g_cm2,
            aspect_ratio,
        )
        return result


# ============================================================
# I/O and plots
# ============================================================
def save_results(
    all_avg_results: Sequence[TDEDynAveragedResult],
    output_dir: Path,
) -> pd.DataFrame:
    """Save averaged results with units encoded in column names."""
    df = pd.DataFrame([r.to_dict() for r in all_avg_results])
    path = output_dir / AVERAGED_RESULTS_FILENAME
    df.to_csv(path, index=False, float_format="%.6e")
    print(f"saved: {path}")
    return df


def save_metadata(
    output_dir: Path,
    config: TDEDynConfig,
    runtime_stats: dict[str, float | int],
) -> None:
    """Save simulation configuration and runtime values with explicit units."""
    metadata: dict[str, Any] = {
        "simulation_info": {
            "date": datetime.now().isoformat(),
            "timestamp_s_since_epoch": time.time(),
            "description": "Dynamical TDE mass-rate simulation with unified parallel damping for primary and copy tasks",
        },
        "parameters": {
            "num_samples": config.num_samp,
            "num_repeats": config.num_repeats,
            "total_simulations": config.num_samp**2 * config.num_repeats,
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
            "nsc_radius_pc_at_reference_mass": (
                config.nsc_radius_pc_at_reference_mass
            ),
            "nsc_radius_reference_mass_msun": (
                NSC_RADIUS_REFERENCE_MASS_MSUN
            ),
            "nsc_radius_mass_slope": config.nsc_radius_mass_slope,
            "dyn_resupply": config.dyn_resupply,
            "max_copies_per_object": config.max_copies_per_object,
            "reuse_same_orbit_for_copies": config.reuse_same_orbit_for_copies,
            "tde_radius_rg": config.tde_radius_rg,
            "direct_tde": config.direct_tde,
            "damping_cpu_count": config.damping_cpu_count,
        },
        "runtime_statistics": runtime_stats,
    }
    path = output_dir / METADATA_FILENAME
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"saved: {path}")


def create_visualizations_from_csv(
    csv_file: Path, output_dir: Path
) -> None:
    """Create plots from unit-labelled averaged simulation columns."""
    try:
        df = pd.read_csv(csv_file)
        if len(df) == 0:
            print("No data to visualize")
            return

        m_smbh_unique_msun = np.sort(df["m_smbh_msun"].unique())
        m_nsc_unique_msun = np.sort(df["m_nsc_msun"].unique())

        rate_2d = df.pivot(
            index="m_smbh_msun",
            columns="m_nsc_msun",
            values="tde_mass_rate_total_mean_msun_per_myr",
        ).values

        fgeo_2d = df.pivot(
            index="m_smbh_msun",
            columns="m_nsc_msun",
            values="f_geo_mean",
        ).values

        # plot 1
        fig1, ax1 = plt.subplots(figsize=(10, 8))
        for m_nsc_msun in m_nsc_unique_msun:
            subset = df[df["m_nsc_msun"] == m_nsc_msun].sort_values(
                "m_smbh_msun"
            )
            ax1.plot(
                subset["m_smbh_msun"],
                subset["tde_mass_rate_total_mean_msun_per_myr"],
                "o-",
                label=f"M_NSC={m_nsc_msun:.1e}",
            )
        ax1.set_xscale("log")
        if np.any(df["tde_mass_rate_total_mean_msun_per_myr"] > 0.0):
            ax1.set_yscale("log")
        ax1.set_xlabel(r"$M_{\rm SMBH}\,[M_\odot]$")
        ax1.set_ylabel(r"TDE mass rate $[M_\odot/{\rm Myr}]$")
        ax1.set_title("Dynamical TDE mass rate vs SMBH mass")
        ax1.grid(True, alpha=0.3, linestyle="--")
        ax1.legend()
        plt.tight_layout()
        plt.savefig(
            output_dir / RATE_VS_SMBH_PLOT_FILENAME,
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig1)

        # plot 2
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        im = ax2.pcolormesh(
            m_smbh_unique_msun,
            m_nsc_unique_msun,
            rate_2d.T,
            shading="auto",
            norm=(
                "log"
                if np.any(np.asarray(rate_2d, dtype=float) > 0.0)
                else None
            ),
            cmap="viridis",
        )
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xlabel(r"$M_{\rm SMBH}\,[M_\odot]$")
        ax2.set_ylabel(r"$M_{\rm NSC}\,[M_\odot]$")
        if m_smbh_unique_msun.size > 1:
            ax2.set_xlim(
                m_smbh_unique_msun.min(), m_smbh_unique_msun.max()
            )
        if m_nsc_unique_msun.size > 1:
            ax2.set_ylim(
                m_nsc_unique_msun.min(), m_nsc_unique_msun.max()
            )
        ax2.set_title(r"Dynamical TDE mass rate $[M_\odot/{\rm Myr}]$")
        plt.colorbar(im)
        plt.tight_layout()
        plt.savefig(
            output_dir / RATE_HEATMAP_FILENAME,
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig2)

        # plot 3
        fig3, ax3 = plt.subplots(figsize=(10, 8))
        im = ax3.pcolormesh(
            m_smbh_unique_msun,
            m_nsc_unique_msun,
            fgeo_2d.T,
            shading="auto",
            cmap="magma",
        )
        ax3.set_xscale("log")
        ax3.set_yscale("log")
        ax3.set_xlabel(r"$M_{\rm SMBH}\,[M_\odot]$")
        ax3.set_ylabel(r"$M_{\rm NSC}\,[M_\odot]$")
        if m_smbh_unique_msun.size > 1:
            ax3.set_xlim(
                m_smbh_unique_msun.min(), m_smbh_unique_msun.max()
            )
        if m_nsc_unique_msun.size > 1:
            ax3.set_ylim(
                m_nsc_unique_msun.min(), m_nsc_unique_msun.max()
            )
        ax3.set_title(r"Geometric selection fraction $f_{\rm geo}$")
        plt.colorbar(im)
        plt.tight_layout()
        plt.savefig(
            output_dir / GEOMETRIC_FRACTION_HEATMAP_FILENAME,
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig3)

        print(f"visualizations saved to: {output_dir}")

    except Exception as e:
        print(f"Warning: visualization failed: {e}")


# ============================================================
# Main
# ============================================================
def print_header(text: str, width: int = 72, char: str = "=") -> None:
    print(f"\n{char * width}")
    print(f"{text:^{width}}")
    print(f"{char * width}")


def debug_print_result(res: TDEDynSingleResult) -> None:
    print("\n========== DEBUG INFO ==========")

    print("\n--- Embedded ---")
    print(f"Stars: {res.embedded_stars_primary}")
    print(f"BH: {res.embedded_bh_primary}")
    print(f"Outside: {res.outside_primary}")

    print("\n--- TDE Counts ---")
    print(f"Primary Direct: {res.direct_tde_primary_count}")
    print(f"Primary Migration: {res.migration_tde_primary_count}")
    print(f"Copy Direct: {res.direct_tde_copy_count}")
    print(f"Copy Migration: {res.migration_tde_copy_count}")

    print("\n--- Mass --- first 5 TDEs ---")
    print(f"Primary masses: {res.tde_masses_primary_msun[:5]}")
    print(f"Copy masses: {res.tde_masses_copies_msun[:5]}")

    print("\n--- Times (Myr) first 5 TDEs ---")
    print(f"Primary times: {res.tde_times_primary_myr[:5]}")
    print(f"Copy times: {res.tde_times_copies_myr[:5]}")

    print("================================\n")


def main(config: Optional[TDEDynConfig] = None) -> pd.DataFrame:
    """Run the configured mass grid and return the averaged result table."""
    if config is None:
        config = TDEDynConfig()

    config.output_dir.mkdir(parents=True, exist_ok=True)

    print_header("TDE DYN SIMULATION", 72, "#")
    print(f"Grid size: {config.num_samp} x {config.num_samp} = {config.num_samp**2}")
    print(f"Repeats per point: {config.num_repeats}")
    print(f"Dynamic resupply: {config.dyn_resupply}")
    print(f"Direct TDE flag: {config.direct_tde}")
    print(f"Output directory: {config.output_dir}")
    print(
        f"Damping CPU count: "
        f"{config.damping_cpu_count if config.damping_cpu_count is not None else 'auto'}"
    )

    m_smbh_samples_msun, m_nsc_samples_msun = config.get_param_grid()
    engine = TDEDynEngine(config)

    all_avg_results: list[TDEDynAveragedResult] = []
    total_simulations = 0
    t0 = time.time()

    for i, m_smbh_msun in enumerate(m_smbh_samples_msun):
        for j, m_nsc_msun in enumerate(m_nsc_samples_msun):
            grid_idx = i * config.num_samp + j + 1
            print_header(
                f"Grid Point {grid_idx}/{config.num_samp**2}: "
                f"M_SMBH={m_smbh_msun:.2e}, M_NSC={m_nsc_msun:.2e}",
                72,
                "-",
            )

            repeat_results: list[TDEDynSingleResult] = []
            for repeat in range(config.num_repeats):
                print(f"Repeat {repeat+1}/{config.num_repeats}")
                res = engine.run_single_simulation(
                    float(m_smbh_msun), float(m_nsc_msun), repeat
                )

                print(f"  total stars          : {res.total_stars:.3e}")
                print(f"  geo-selected stars   : {res.geo_selected_stars:.3e}")
                print(f"  f_geo                : {res.f_geo:.3e}")
                print(
                    "  primary TDE mass     : "
                    f"{res.tde_mass_primary_msun:.3e} Msun"
                )
                print(
                    f"  copy TDE mass        : "
                    f"{res.copy_tde_mass_msun:.3e} Msun"
                )
                print(
                    f"  total TDE mass       : "
                    f"{res.tde_mass_total_msun:.3e} Msun"
                )
                print(
                    "  total TDE mass rate  : "
                    f"{res.tde_mass_rate_total_msun_per_myr:.3e} Msun/Myr"
                )
                print(f"  copies generated     : {res.num_copies_generated:.3e}")
                print(f"  total TDE count      : {res.tde_star_count_total:.3e}")

                repeat_results.append(res)
                debug_print_result(res)

                total_simulations += 1

            all_avg_results.append(
                TDEDynAveragedResult(
                    float(m_smbh_msun),
                    float(m_nsc_msun),
                    repeat_results,
                )
            )

            if (grid_idx % 5 == 0) or (grid_idx == config.num_samp**2):
                save_results(all_avg_results, config.output_dir)

    total_runtime = time.time() - t0

    print_header("SIMULATION COMPLETE", 72, "#")
    print(f"Total simulations: {total_simulations}")
    print(
        f"Runtime: {total_runtime:.2f} s "
        f"({total_runtime/SECONDS_PER_MINUTE:.2f} min)"
    )

    df = save_results(all_avg_results, config.output_dir)
    runtime_stats: dict[str, float | int] = {
        "total_simulations": total_simulations,
        "total_runtime_seconds": total_runtime,
        "total_runtime_minutes": total_runtime / SECONDS_PER_MINUTE,
        "average_time_per_simulation_seconds": (
            total_runtime / total_simulations if total_simulations > 0 else np.nan
        ),
    }
    save_metadata(config.output_dir, config, runtime_stats)

    viz_dir = config.output_dir / VISUALIZATION_DIRECTORY
    viz_dir.mkdir(parents=True, exist_ok=True)
    create_visualizations_from_csv(
        config.output_dir / AVERAGED_RESULTS_FILENAME,
        viz_dir,
    )

    return df


def run_cli(argv: Optional[Sequence[str]] = None) -> None:
    """Run the command-line interface using ``argv`` without the program name."""
    cli_args = list(sys.argv[1:] if argv is None else argv)
    num_samples = int(cli_args[0]) if len(cli_args) > 0 else 5
    num_repeats = int(cli_args[1]) if len(cli_args) > 1 else 3
    output_dir = (
        Path(cli_args[2])
        if len(cli_args) > 2
        else Path("./tde_dyn_results")
    )
    damping_cpu_count = int(cli_args[3]) if len(cli_args) > 3 else None

    config = TDEDynConfig(
        num_samp=num_samples,
        num_repeats=num_repeats,
        output_dir=output_dir,
        dyn_resupply=True,
        max_copies_per_object=30,
        reuse_same_orbit_for_copies=True,
        direct_tde=False,
        damping_cpu_count=damping_cpu_count,
    )

    time0 = time.time()
    main(config)
    time1 = time.time()
    print(
        f"Total runtime: {time1 - time0:.2f} seconds = "
        f"{((time1 - time0)/SECONDS_PER_MINUTE):.2f} minutes"
    )


if __name__ == "__main__":
    run_cli()
