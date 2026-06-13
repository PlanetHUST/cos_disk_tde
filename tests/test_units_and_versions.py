"""Regression checks for unit contracts and the consolidated entry point."""

import unittest
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

import constants as const
import helper_function as helpers
import main
import sampling
import test
from general_class import AGN_properties


class UnitContractTests(unittest.TestCase):
    def test_main_reexports_canonical_simulation(self) -> None:
        self.assertIs(main.TDEDynConfig, test.TDEDynConfig)
        self.assertIs(main.TDEDynEngine, test.TDEDynEngine)

    def test_project_constant_aliases_are_consistent(self) -> None:
        self.assertEqual(const.Myr, const.MEGAYEAR_S)
        self.assertEqual(const.msun, const.SOLAR_MASS_G)
        self.assertEqual(const.parsec, const.PARSEC_CM)

    def test_mass_grid_is_in_solar_masses(self) -> None:
        masses_msun = helpers.initial_mass(3, 1.0e4, 1.0e6)
        np.testing.assert_allclose(masses_msun, [1.0e4, 1.0e5, 1.0e6])

    def test_orbital_samples_have_consistent_shapes(self) -> None:
        np.random.seed(7)
        orbital_values = helpers.initial_star(
            num=8, M_SMBH=1.0e7, rmax=1.0e18
        )
        for values in orbital_values:
            self.assertEqual(values.shape, (8,))

    def test_agn_properties_use_scalar_cgs_values(self) -> None:
        agn = AGN_properties()
        agn.init_AGN(1.0e7, 10.0)
        self.assertEqual(agn.M_SMBH, 1.0e7)
        self.assertEqual(agn.AGNlifetime, 10.0)
        self.assertGreater(agn.rg, 0.0)
        self.assertGreater(agn.sigma, 0.0)

    def test_imf_sampler_returns_positive_masses_msun(self) -> None:
        np.random.seed(11)
        masses_msun = sampling.sample_salpeter(16)
        self.assertEqual(masses_msun.shape, (16,))
        self.assertTrue(np.all(masses_msun > 0.0))

    def test_geometry_returns_radii_in_input_length_unit(self) -> None:
        engine = test.TDEDynEngine(test.TDEDynConfig())
        mask, inner_radius_cm, outer_radius_cm = (
            engine.select_disk_crossing_stars(
                semi_major_axis_cm=np.array([10.0, 100.0]),
                eccentricity=np.zeros(2),
                cos_omega=np.zeros(2),
                disk_inner_radius_cm=5.0,
                disk_outer_radius_cm=50.0,
            )
        )
        self.assertEqual(mask.tolist(), [True, False])
        np.testing.assert_allclose(inner_radius_cm, [10.0, 100.0])
        np.testing.assert_allclose(outer_radius_cm, [10.0, 100.0])

    def test_zero_rate_single_grid_visualization_is_valid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            csv_path = output_dir / test.AVERAGED_RESULTS_FILENAME
            plot_dir = output_dir / test.VISUALIZATION_DIRECTORY
            plot_dir.mkdir()
            pd.DataFrame(
                [
                    {
                        "m_smbh_msun": 1.0e4,
                        "m_nsc_msun": 1.0e2,
                        "tde_mass_rate_total_mean_msun_per_myr": 0.0,
                        "f_geo_mean": 0.0,
                    }
                ]
            ).to_csv(csv_path, index=False)
            test.create_visualizations_from_csv(csv_path, plot_dir)
            self.assertTrue(
                (plot_dir / test.RATE_HEATMAP_FILENAME).exists()
            )


if __name__ == "__main__":
    unittest.main()
