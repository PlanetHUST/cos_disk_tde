"""Regression checks for CLI parsing and root wrapper behavior."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import main
import test
from src import tde_rate


class CliEntrypointTests(unittest.TestCase):
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

    def test_cli_accepts_explicit_mass_vectors_and_exposes_physics(self) -> None:
        arguments = tde_rate.build_parser().parse_args(
            ["4", "/tmp/results", "3", "--smbh-masses", "1e3", "3e3",
             "--nsc-masses", "1e4", "2e5", "--no-resupply"]
        )
        self.assertEqual(arguments.num_repeats, 4)
        self.assertEqual(arguments.smbh_masses, [1e3, 3e3])
        self.assertEqual(arguments.nsc_masses, [1e4, 2e5])
        self.assertEqual(arguments.damping_cpu_count, 3)
        self.assertTrue(arguments.no_resupply)
        self.assertEqual(arguments.duty_cycle_myr, 10.0)

    def test_cli_can_regenerate_all_plots_without_simulating(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            pd.DataFrame(
                [
                    {
                        "m_smbh_msun": 1.0e4,
                        "m_nsc_msun": 1.0e5,
                        "repeat": 0,
                        "task_kind": "primary",
                        "event_time_myr": 1.0,
                        "event_mass_msun": 1.0,
                        "cumulative_tde_mass_msun": 1.0,
                    },
                    {
                        "m_smbh_msun": 1.0e4,
                        "m_nsc_msun": 1.0e5,
                        "repeat": 0,
                        "task_kind": "copy",
                        "event_time_myr": 2.0,
                        "event_mass_msun": 0.5,
                        "cumulative_tde_mass_msun": 1.5,
                    },
                ]
            ).to_csv(
                output_dir / tde_rate.TIMESCALES_FILENAME,
                index=False,
            )
            pd.DataFrame(
                [
                    {
                        "m_smbh_msun": 1.0e4,
                        "m_nsc_msun": 1.0e5,
                        "tde_mass_rate_total_mean_msun_per_myr": 1.0,
                        "f_geo_mean": 0.1,
                    }
                ]
            ).to_csv(
                output_dir / tde_rate.AVERAGED_RESULTS_FILENAME,
                index=False,
            )
            pd.DataFrame(
                [
                    {
                        "m_smbh_msun": 1.0e4,
                        "m_nsc_msun": 1.0e5,
                        "repeat": 0,
                        "task_kind": "primary",
                        "semi_major_axis_cm": 1.0e16,
                        "eccentricity": 0.1,
                        "inclination_rad": 0.01,
                        "mass_msun": 1.0,
                    }
                ]
            ).to_csv(output_dir / "orbital_params.csv", index=False)

            with patch("src.tde_rate.cli.run_simulation") as simulation:
                tde_rate.run_cli(["--plot-only", str(output_dir)])

            simulation.assert_not_called()
            plot_dir = output_dir / tde_rate.VISUALIZATION_DIRECTORY
            self.assertTrue((plot_dir / tde_rate.RATE_VS_SMBH_PLOT_FILENAME).exists())
            self.assertTrue((plot_dir / tde_rate.RATE_HEATMAP_FILENAME).exists())
            self.assertTrue(
                (plot_dir / tde_rate.PRIMARY_TIMESCALE_HISTOGRAM_FILENAME).exists()
            )
            self.assertTrue(
                (plot_dir / tde_rate.COMBINED_TIMESCALE_HISTOGRAM_FILENAME).exists()
            )
            self.assertTrue(
                (plot_dir / tde_rate.COPY_TIMESCALE_HISTOGRAM_FILENAME).exists()
            )
            for parameter in ("sma", "ecc", "inc", "mass"):
                self.assertTrue((plot_dir / f"tde_pdf_{parameter}.png").exists())
                self.assertTrue((plot_dir / f"tde_hist_{parameter}.png").exists())


if __name__ == "__main__":
    unittest.main()
