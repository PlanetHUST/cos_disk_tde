"""Regression checks for result persistence and visualization outputs."""

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

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
        single_result.tde_times_primary_myr.append(1.0)
        single_result.tde_masses_primary_msun.append(1.0)
        single_result.tde_times_copies_myr.append(3.0)
        single_result.tde_masses_copies_msun.append(0.5)
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

            self.assertEqual(
                list(frame.columns),
                [
                    "m_smbh_msun",
                    "m_nsc_msun",
                    "repeat",
                    "task_kind",
                    "event_time_myr",
                    "event_mass_msun",
                    "cumulative_tde_mass_msun",
                ],
            )
            self.assertEqual(len(statistics), 3)
            self.assertEqual(
                list(statistics.columns),
                ["task_kind", "event_count", "tde_mass_msun"],
            )
            self.assertEqual(
                statistics.loc[
                    statistics["task_kind"] == "primary",
                    "event_count",
                ].iloc[0],
                1,
            )
            self.assertEqual(
                statistics.loc[
                    statistics["task_kind"] == "total",
                    "tde_mass_msun",
                ].iloc[0],
                1.5,
            )
            self.assertIn("TDE event statistics", printed.getvalue())
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
                        "m_smbh_msun": 1.0e4,
                        "m_nsc_msun": 1.0e5,
                        "repeat": 0,
                        "task_kind": "primary",
                        "event_time_myr": 0.0,
                        "event_mass_msun": 0.0,
                        "cumulative_tde_mass_msun": 0.0,
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
