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
from src import tde_rate
from src.tde_rate import workers


class UnitContractTests(unittest.TestCase):
    def test_main_reexports_canonical_simulation(self) -> None:
        self.assertIs(main.TDEDynConfig, tde_rate.TDEDynConfig)
        self.assertIs(test.TDEDynEngine, tde_rate.TDEDynEngine)
        self.assertIs(main.TDEDynEngine, test.TDEDynEngine)

    def test_root_scripts_only_delegate_to_src_package(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        for script_name in ("main.py", "test.py"):
            script_text = (project_root / script_name).read_text()
            self.assertIn("from src.tde_rate import", script_text)
            self.assertLess(len(script_text.splitlines()), 20)

    def test_cli_preserves_positional_defaults_and_exposes_physics(self) -> None:
        arguments = tde_rate.build_parser().parse_args(
            ["2", "4", "/tmp/results", "3", "--no-resupply"]
        )
        self.assertEqual(arguments.num_samples, 2)
        self.assertEqual(arguments.num_repeats, 4)
        self.assertEqual(arguments.damping_cpu_count, 3)
        self.assertTrue(arguments.no_resupply)
        self.assertEqual(arguments.duty_cycle_myr, 10.0)

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

    def test_copy_damping_time_excludes_relaxation_offset(self) -> None:
        class FakeCalculator:
            def damping_simulation(
                self,
                mass_g: float,
                state: object,
                formation_time_myr: float,
                object_name: str,
                type_flag: int,
            ) -> tuple[float, float, int]:
                return 3.0 * const.Myr, 2.0e15, 0

        original_calculator = workers._WORKER_CALCULATOR
        workers._WORKER_CALCULATOR = FakeCalculator()
        try:
            result = workers.process_damping_task(
                {
                    "task_kind": "copy",
                    "idx": 2,
                    "origin_primary_idx": 0,
                    "copy_number": 1,
                    "mass_g": const.msun,
                    "semi_major_axis_cm": 3.0e15,
                    "eccentricity": 0.2,
                    "inclination_rad": 0.1,
                    "cos_omega": 0.5,
                    "formation_time_myr": 5.0,
                    "state_time_s": const.Myr,
                    "type_flag": 0,
                }
            )
        finally:
            workers._WORKER_CALCULATOR = original_calculator

        self.assertEqual(result["state_time_myr"], 1.0)
        self.assertEqual(result["final_time_myr"], 3.0)
        self.assertEqual(result["damping_time_myr"], 2.0)

    def test_timescale_csv_statistics_and_histograms(self) -> None:
        single_result = tde_rate.TDEDynSingleResult(1.0e7, 1.0e6, 0)
        single_result.timescale_records.append(
            {
                "m_smbh_msun": 1.0e7,
                "m_nsc_msun": 1.0e6,
                "repeat": 0,
                "task_kind": "primary",
                "object_index": 0,
                "origin_primary_index": -1,
                "copy_number": -1,
                "object_type_flag": 0,
                "mass_msun": 1.0,
                "initial_semi_major_axis_cm": 2.0e16,
                "final_semi_major_axis_cm": 1.0e16,
                "eccentricity": 0.2,
                "inclination_rad": 0.1,
                "state_time_myr": 0.0,
                "formation_time_myr": 4.0,
                "damping_time_myr": 0.25,
                "migration_time_myr": 0.75,
                "relaxation_time_myr": 2.0,
                "delivery_time_myr": 1.0,
                "outcome": "migration_tde",
                "counted_as_tde": True,
                "tde_mode": "migration",
            }
        )
        averaged_result = tde_rate.TDEDynAveragedResult(
            1.0e7, 1.0e6, [single_result]
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            plot_dir = output_dir / "plots"
            frame = tde_rate.save_timescales(
                [averaged_result], output_dir
            )
            csv_path = output_dir / tde_rate.TIMESCALES_FILENAME
            statistics = tde_rate.print_timescale_statistics(csv_path)
            tde_rate.create_timescale_histograms_from_csv(
                csv_path, plot_dir
            )

            self.assertEqual(
                list(frame.columns), list(tde_rate.TIMESCALE_COLUMNS)
            )
            self.assertEqual(len(statistics), 4)
            self.assertEqual(
                statistics.loc[
                    statistics["timescale"] == "damping_time_myr",
                    "median_myr",
                ].iloc[0],
                0.25,
            )
            self.assertTrue(
                (
                    plot_dir
                    / tde_rate.TIMESCALE_HISTOGRAM_FILENAME
                ).exists()
            )


if __name__ == "__main__":
    unittest.main()
