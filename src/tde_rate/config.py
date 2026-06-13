"""Configuration and model constants for the TDE-rate workflow."""

from pathlib import Path

import numpy as np
import numpy.typing as npt

import helper_function as helpers

FloatArray = npt.NDArray[np.float64]

YEARS_PER_MYR = 1.0e6
SECONDS_PER_MINUTE = 60.0
DISK_INNER_RADIUS_RG = 6.0
DISK_ACCRETION_RATE_G_S_PER_MSUN = 1.44e18
NSC_RADIUS_REFERENCE_MASS_MSUN = 3.6e6
NSC_RADIUS_LOG10_OFFSET = -0.011
MIN_RELAXATION_TIME_MYR = 1.0e-12

AVERAGED_RESULTS_FILENAME = "tde_dyn_averaged_results.csv"
METADATA_FILENAME = "simulation_metadata.json"
TIMESCALES_FILENAME = "timescales.csv"
VISUALIZATION_DIRECTORY = "visualizations"
RATE_VS_SMBH_PLOT_FILENAME = "tde_dyn_rate_vs_smbh.png"
RATE_HEATMAP_FILENAME = "tde_dyn_rate_heatmap.png"
GEOMETRIC_FRACTION_HEATMAP_FILENAME = "tde_dyn_fgeo_heatmap.png"
TIMESCALE_HISTOGRAM_FILENAME = "timescale_histograms.png"


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
        damping_cpu_count: int | None = None,
        checkpoint_grid_interval: int = 5,
        create_visualizations: bool = True,
        print_debug_results: bool = True,
    ) -> None:
        if num_samp <= 0 or num_repeats <= 0:
            raise ValueError("sample and repeat counts must be positive")
        if duty_cycle_myr <= 0.0:
            raise ValueError("AGN duty cycle must be positive")
        if checkpoint_grid_interval <= 0:
            raise ValueError("checkpoint interval must be positive")
        self.num_samp = num_samp
        self.num_repeats = num_repeats
        self.duty_cycle_myr = duty_cycle_myr
        self.m_smbh_min_msun = m_smbh_min_msun
        self.m_smbh_max_msun = m_smbh_max_msun
        self.m_nsc_min_msun = m_nsc_min_msun
        self.m_nsc_max_msun = m_nsc_max_msun
        self.output_dir = Path(output_dir)
        self.metallicity = metallicity
        self.nsc_radius_pc_at_reference_mass = nsc_radius_pc_at_reference_mass
        self.nsc_radius_mass_slope = nsc_radius_mass_slope
        self.use_disk_inner_radius = use_disk_inner_radius
        self.dyn_resupply = dyn_resupply
        self.max_copies_per_object = max_copies_per_object
        self.reuse_same_orbit_for_copies = reuse_same_orbit_for_copies
        self.tde_radius_rg = tde_radius_rg
        self.direct_tde = direct_tde
        self.damping_cpu_count = damping_cpu_count
        self.checkpoint_grid_interval = checkpoint_grid_interval
        self.create_visualizations = create_visualizations
        self.print_debug_results = print_debug_results

    def get_param_grid(self) -> tuple[FloatArray, FloatArray]:
        """Return logarithmic SMBH and NSC mass grids [Msun]."""
        return (
            np.asarray(
                helpers.initial_mass(
                    self.num_samp,
                    self.m_smbh_min_msun,
                    self.m_smbh_max_msun,
                ),
                dtype=float,
            ),
            np.asarray(
                helpers.initial_mass(
                    self.num_samp,
                    self.m_nsc_min_msun,
                    self.m_nsc_max_msun,
                ),
                dtype=float,
            ),
        )
