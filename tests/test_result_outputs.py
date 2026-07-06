"""Regression checks for result persistence and visualization outputs."""

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

from src import tde_rate


class OutputTests(unittest.TestCase):
    def test_zero_rate_single_grid_visualization_is_valid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            csv_path = output_dir / tde_rate.AVERAGED_RESULTS_FILENAME
            plot_dir = output_dir / tde_rate.VISUALIZATION_DIRECTORY
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
            tde_rate.create_visualizations_from_csv(csv_path, plot_dir)
            self.assertTrue((plot_dir / tde_rate.RATE_HEATMAP_FILENAME).exists())

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
        averaged_result = tde_rate.TDEDynAveragedResult(1.0e7, 1.0e6, [single_result])

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            plot_dir = output_dir / "plots"
            frame = tde_rate.save_timescales([averaged_result], output_dir)
            csv_path = output_dir / tde_rate.TIMESCALES_FILENAME
            printed = io.StringIO()
            with redirect_stdout(printed):
                statistics = tde_rate.print_timescale_statistics(csv_path)
            tde_rate.create_timescale_histograms_from_csv(csv_path, plot_dir)

            self.assertEqual(list(frame.columns), list(tde_rate.TIMESCALE_COLUMNS))
            self.assertEqual(len(statistics), 4)
            self.assertIn("std_myr", statistics.columns)
            self.assertIn("p25_myr", statistics.columns)
            self.assertIn("positive_count", statistics.columns)
            self.assertEqual(
                statistics.loc[
                    statistics["timescale"] == "damping_time_myr",
                    "median_myr",
                ].iloc[0],
                0.25,
            )
            self.assertIn(
                "Timescale statistics by task kind",
                printed.getvalue(),
            )
            self.assertIn("Timescale outcome counts", printed.getvalue())
            self.assertTrue((plot_dir / tde_rate.TIMESCALE_HISTOGRAM_FILENAME).exists())
            self.assertTrue(
                (plot_dir / tde_rate.COMBINED_TIMESCALE_HISTOGRAM_FILENAME).exists()
            )
            self.assertTrue(
                (plot_dir / tde_rate.COPY_TIMESCALE_HISTOGRAM_FILENAME).exists()
            )

    def test_timescale_histograms_allow_no_positive_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            csv_path = output_dir / tde_rate.TIMESCALES_FILENAME
            pd.DataFrame(
                [
                    {
                        "task_kind": "primary",
                        "damping_time_myr": 0.0,
                        "migration_time_myr": np.nan,
                        "relaxation_time_myr": -1.0,
                        "delivery_time_myr": 0.0,
                    }
                ]
            ).to_csv(csv_path, index=False)

            tde_rate.create_timescale_histograms_from_csv(
                csv_path,
                output_dir,
            )

            self.assertTrue(
                (output_dir / tde_rate.PRIMARY_TIMESCALE_HISTOGRAM_FILENAME).exists()
            )
            self.assertTrue(
                (output_dir / tde_rate.COPY_TIMESCALE_HISTOGRAM_FILENAME).exists()
            )
            self.assertTrue(
                (output_dir / tde_rate.COMBINED_TIMESCALE_HISTOGRAM_FILENAME).exists()
            )


if __name__ == "__main__":
    unittest.main()
