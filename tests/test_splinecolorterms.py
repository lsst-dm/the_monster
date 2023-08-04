import unittest
import numpy as np
import astropy.units as units
from scipy.stats import median_abs_deviation

import lsst.utils

from lsst.the.monster import ColortermSplineFitter


class MonsterColortermSplineFitterTest(lsst.utils.tests.TestCase):
    def check_splinefitter(self, mag_offset=0.0, check_pars=True):
        """Check the spline fitter, with optional magnitude offset.

        Parameters
        ----------
        mag_offset : `float`, optional
            Overall magnitude offset between ref and meas catalogs.
        check_pars : `bool`, optional
            Check the parameter values? (Only works well if mag_offset=0.0).
        """
        n_star = 20_000
        n_nodes = 10

        # Create some stars of some colors, scatter, outliers.
        # These are uniform for simplicity.
        colors = np.random.uniform(0.5, 3.5, n_star)

        mag_ref = np.random.uniform(17.0, 21.0, n_star)

        nodes = np.linspace(colors.min(), colors.max(), n_nodes)

        node_values = np.array([-0.2, -0.1, 0.0, 0.1, 0.2, 0.1, 0.15, 0.1, 0.0, -0.1]) + 0.02

        spl = lsst.afw.math.makeInterpolate(
            nodes,
            node_values,
            lsst.afw.math.stringToInterpStyle("CUBIC_SPLINE"),
        )

        mag_meas = mag_ref + spl.interpolate(colors)

        mag_ref_scatter = mag_ref + np.random.normal(loc=0.0, scale=0.02, size=n_star)
        mag_meas_scatter = mag_meas + np.random.normal(loc=0.0, scale=0.02, size=n_star) + mag_offset

        # Add a couple of large outliers
        outlier_indices = np.hstack((np.arange(10), np.arange(10) + 100))
        mag_ref_scatter[0: 10] = 50.0
        mag_meas_scatter[100: 110] = 50.0
        non_outliers = np.ones(n_star, dtype=bool)
        non_outliers[0: 10] = False
        non_outliers[100: 110] = False

        flux_ref_scatter = (mag_ref_scatter*units.ABmag).to_value(units.nJy)
        flux_meas_scatter = (mag_meas_scatter*units.ABmag).to_value(units.nJy)

        # Test the fitter.
        fitter = ColortermSplineFitter(
            colors,
            flux_ref_scatter,
            flux_meas_scatter,
            nodes,
            fit_flux_offset=False,
        )
        p0 = fitter.estimate_p0()
        pars = fitter.fit(p0)

        if check_pars:
            # Convert magnitude nodes to flux.
            node_values_flux = 1.0 - node_values/1.086

            # This is not as close as I'd like, but it does yield good values.
            self.assertFloatsAlmostEqual(pars, node_values_flux, rtol=0.025)

        # Create the spline and apply it to correct the flux.
        spl = lsst.afw.math.makeInterpolate(
            nodes,
            pars,
            lsst.afw.math.stringToInterpStyle("CUBIC_SPLINE"),
        )
        flux_meas_scatter_corr = flux_meas_scatter / spl.interpolate(colors)

        # The tests are based on the ratio of corrected measured flux to
        # reference flux.
        ratio = flux_meas_scatter_corr / flux_ref_scatter

        ratio_med = np.median(ratio[non_outliers])
        ratio_sig = median_abs_deviation(ratio[non_outliers], scale="normal")

        # Somewhat arbitrary comparisons
        self.assertFloatsAlmostEqual(ratio_med, 1.0, rtol=1e-3)
        self.assertLess(ratio_sig, 0.03)

        n_4sig = (np.abs(ratio[non_outliers] - 1.0) > 4.0*ratio_sig).sum()
        self.assertLess(n_4sig, 3e-4*n_star)

    def test_splinefitter(self):
        np.random.seed(1234)
        self.check_splinefitter()

    def test_splinefitter_fluxoffset(self):
        np.random.seed(4321)
        self.check_splinefitter(mag_offset=0.3, check_pars=False)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
