"""Regression checks for low-level unit and sampling contracts."""

import unittest

import numpy as np

import constants as const
import helper_function as helpers
import sampling
from general_class import AGN_properties
from src import tde_rate


class UnitContractTests(unittest.TestCase):
    def test_project_constant_aliases_are_consistent(self) -> None:
        self.assertEqual(const.Myr, const.MEGAYEAR_S)
        self.assertEqual(const.msun, const.SOLAR_MASS_G)
        self.assertEqual(const.parsec, const.PARSEC_CM)

    def test_mass_grid_is_in_solar_masses(self) -> None:
        masses_msun = helpers.initial_mass(3, 1.0e4, 1.0e6)
        np.testing.assert_allclose(masses_msun, [1.0e4, 1.0e5, 1.0e6])

    def test_config_preserves_explicit_rectangular_mass_grid(self) -> None:
        config = tde_rate.TDEDynConfig(
            smbh_masses_msun=[1e3, 3e3], nsc_masses_msun=[1e4, 1e5, 2e5]
        )
        smbh_masses, nsc_masses = config.get_param_grid()
        np.testing.assert_array_equal(smbh_masses, [1e3, 3e3])
        np.testing.assert_array_equal(nsc_masses, [1e4, 1e5, 2e5])

    def test_orbital_samples_have_consistent_shapes(self) -> None:
        np.random.seed(7)
        orbital_values = helpers.initial_star(num=8, M_SMBH=1.0e7, rmax=1.0e18)
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


if __name__ == "__main__":
    unittest.main()
