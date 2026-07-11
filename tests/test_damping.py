"""Regression checks for geometric selection and damping classification."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

import constants as const
import helper_function as helpers
from general_class import AGN_properties
from src import tde_rate
from src.tde_rate import workers


def print_single_star_diagnostics(
    damping_result: dict[str, float | int | str],
    record: dict[str, float | int | str | bool],
    gravitational_radius_cm: float,
    tde_radius_cm: float,
) -> None:
    """Print the deterministic single-star damping diagnostic."""
    initial_axis_cm = float(record["initial_semi_major_axis_cm"])
    final_axis_cm = float(record["final_semi_major_axis_cm"])
    timescale_values = np.array(
        [
            record["damping_time_myr"],
            record["migration_time_myr"],
            record["delivery_time_myr"],
            record["relaxation_time_myr"],
        ],
        dtype=float,
    )
    print("\nSingle-star damping diagnostic")
    print("--------------------------------")
    print(f"outcome              : {record['outcome']}")
    print(f"counted_as_tde       : {record['counted_as_tde']}")
    print(f"tde_mode             : {record['tde_mode']}")
    print(f"object_type_flag     : {record['object_type_flag']}")
    print(f"mass                 : {record['mass_msun']:.6e} Msun")
    print(f"M_SMBH               : {record['m_smbh_msun']:.6e} Msun")
    print(f"M_NSC                : {record['m_nsc_msun']:.6e} Msun")
    print(f"r_g                  : {gravitational_radius_cm:.6e} cm")
    print(
        "initial_a            : "
        f"{initial_axis_cm:.6e} cm = "
        f"{initial_axis_cm / gravitational_radius_cm:.6e} r_g"
    )
    print(
        "final_a              : "
        f"{final_axis_cm:.6e} cm = "
        f"{final_axis_cm / gravitational_radius_cm:.6e} r_g"
    )
    print(
        "tde_radius           : "
        f"{tde_radius_cm:.6e} cm = "
        f"{tde_radius_cm / gravitational_radius_cm:.6e} r_g"
    )
    print(f"eccentricity         : {record['eccentricity']:.6e}")
    print(f"inclination          : {record['inclination_rad']:.6e} rad")
    print(f"cos_omega            : {damping_result['cos_omega']:.6e}")
    print(f"tdamp                : {record['damping_time_myr']:.6e} Myr")
    print(f"tmig                 : {record['migration_time_myr']:.6e} Myr")
    print(f"trelax               : {record['relaxation_time_myr']:.6e} Myr")
    print(f"tdelivery            : {record['delivery_time_myr']:.6e} Myr")
    print(
        "timescale stats      : "
        f"min={np.min(timescale_values):.6e}, "
        f"median={np.median(timescale_values):.6e}, "
        f"mean={np.mean(timescale_values):.6e}, "
        f"max={np.max(timescale_values):.6e} Myr"
    )


class DampTest(unittest.TestCase):
    def test_geometry_unit(self) -> None:
        engine = tde_rate.TDEDynEngine(tde_rate.TDEDynConfig())
        mask, inner_radius_cm, outer_radius_cm = engine.select_disk_crossing_stars(
            semi_major_axis_cm=np.array([10.0, 100.0]),
            eccentricity=np.zeros(2),
            cos_omega=np.zeros(2),
            disk_inner_radius_cm=5.0,
            disk_outer_radius_cm=50.0,
        )
        self.assertEqual(mask.tolist(), [True, False])
        np.testing.assert_allclose(inner_radius_cm, [10.0, 100.0])
        np.testing.assert_allclose(outer_radius_cm, [10.0, 100.0])

    def test_nsc_radius_model_uses_config_values(self) -> None:
        config = tde_rate.TDEDynConfig(
            nsc_radius_pc_at_reference_mass=2.0,
            nsc_radius_reference_mass_msun=4.0,
            nsc_radius_mass_slope=0.5,
        )
        engine = tde_rate.TDEDynEngine(config)

        radius_cm = engine.estimate_nsc_radius_cm(16.0)

        self.assertAlmostEqual(radius_cm, 4.0 * const.parsec)

    def test_time_offset(self) -> None:
        damping_simulation = Mock(return_value=(3.0 * const.Myr, 2.0e15, 0))
        original_calculator = workers._WORKER_CALCULATOR
        workers._WORKER_CALCULATOR = SimpleNamespace(
            damping_simulation=damping_simulation
        )
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
        damping_simulation.assert_called_once()

    def test_single_typical_star(
        self,
    ) -> None:
        tde_rate.set_verbose(False)
        self.addCleanup(tde_rate.set_verbose, True)
        config = tde_rate.TDEDynConfig(
            duty_cycle_myr=10.0,
            dyn_resupply=False,
            direct_tde=False,
            damping_cpu_count=1,
            print_debug_results=False,
        )
        engine = tde_rate.TDEDynEngine(config)
        agn_properties = AGN_properties()
        agn_properties.init_AGN(1.0e4, config.duty_cycle_myr)
        disk_data = engine.get_disk_profiles(agn_properties.M_SMBH)

        original_calculator = workers._WORKER_CALCULATOR
        workers.init_damping_worker(
            agn_properties,
            disk_data["radius_cm"],
            disk_data["surface_density_g_cm2"],
            disk_data["aspect_ratio"],
        )
        try:
            damping_result = workers.process_damping_task(
                {
                    "task_kind": "primary",
                    "idx": 0,
                    "origin_primary_idx": -1,
                    "copy_number": -1,
                    "mass_g": const.msun,
                    "semi_major_axis_cm": 5.0e16,
                    "eccentricity": 0.5,
                    "inclination_rad": 0.2,
                    "cos_omega": 0.0,
                    "formation_time_myr": 1000.0,
                    "state_time_s": 0.0,
                    "type_flag": 0,
                }
            )
        finally:
            workers._WORKER_CALCULATOR = original_calculator

        single_result = tde_rate.TDEDynSingleResult(
            agn_properties.M_SMBH, 1.0e5, 0
        )
        relaxation_time_myr = np.array(
            [
                helpers.t_relax(
                    agn_properties.M_SMBH,
                    damping_result["initial_semi_major_axis_cm"],
                )
                / const.Myr
            ]
        )
        engine.classify_primary_and_copies(
            single_result,
            agn_properties,
            [damping_result],
            [],
            relaxation_time_myr,
            disk_data["radius_cm"],
            disk_data["surface_density_g_cm2"],
            disk_data["aspect_ratio"],
        )
        record = single_result.timescale_records[0]
        print_single_star_diagnostics(
            damping_result,
            record,
            agn_properties.rg,
            config.tde_radius_rg * agn_properties.rg,
        )

        self.assertEqual(damping_result["obj_type"], 0)
        self.assertEqual(single_result.embedded_stars_primary, 1)
        self.assertEqual(single_result.migration_tde_primary_count, 1)
        self.assertEqual(single_result.tde_star_count_total, 1)
        self.assertEqual(record["outcome"], "migration_tde")
        self.assertTrue(record["counted_as_tde"])
        self.assertEqual(record["tde_mode"], "migration")
        np.testing.assert_allclose(
            record["damping_time_myr"],
            0.28721470119652454,
            rtol=1.0e-6,
        )
        np.testing.assert_allclose(
            record["migration_time_myr"],
            0.27274392867221375,
            rtol=1.0e-6,
        )
        np.testing.assert_allclose(
            record["delivery_time_myr"],
            0.5599586298687382,
            rtol=1.0e-6,
        )
        np.testing.assert_allclose(
            record["final_semi_major_axis_cm"],
            4.2105683847940344e16,
            rtol=1.0e-6,
        )


if __name__ == "__main__":
    unittest.main()
