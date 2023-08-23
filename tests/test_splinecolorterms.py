import os
import unittest
import numpy as np
import astropy.units as units
from scipy.stats import median_abs_deviation
import tempfile

import lsst.utils

from lsst.the.monster import ColortermSplineFitter, ColortermSpline


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


class MonsterColortermSplineTest(lsst.utils.tests.TestCase):
    def setUp(self):
        self._source_survey = "Survey1"
        self._target_survey = "Survey2"
        self._source_color_field_1 = "flux_g"
        self._source_color_field_2 = "flux_i"
        self._source_field = "flux_r"
        self._nodes = np.linspace(0.5, 3.5, 10)
        self._values = np.ones(10)
        self._values[4: 6] = 1.1
        self._flux_offset = 1.0

    def test_spline_apply(self):
        np.random.seed(12345)

        spline = ColortermSpline(
            self._source_survey,
            self._target_survey,
            self._source_color_field_1,
            self._source_color_field_2,
            self._source_field,
            self._nodes,
            self._values,
            flux_offset=self._flux_offset,
        )

        n_star = 20_000

        colors = np.random.uniform(0.5, 3.5, n_star)

        flux_1 = np.ones(n_star)
        mag_1 = (flux_1*units.nJy).to_value(units.ABmag)
        mag_2 = mag_1 - colors
        flux_2 = (mag_2*units.ABmag).to_value(units.nJy)

        flux_source = np.zeros(n_star) + 10000.0

        model_flux = spline.apply(flux_1, flux_2, flux_source)

        # And compare directly.
        spl = lsst.afw.math.makeInterpolate(
            self._nodes,
            self._values,
            lsst.afw.math.stringToInterpStyle("CUBIC_SPLINE"),
        )
        model = spl.interpolate(colors)
        model -= self._flux_offset/flux_source

        np.testing.assert_array_almost_equal(model_flux, model)

    def test_spline_apply_out_of_bounds(self):
        np.random.seed(12345)

        spline = ColortermSpline(
            self._source_survey,
            self._target_survey,
            self._source_color_field_1,
            self._source_color_field_2,
            self._source_field,
            self._nodes,
            self._values,
            flux_offset=self._flux_offset,
        )

        n_star = 1_000
        colors = np.zeros(n_star)
        colors[0: n_star // 2] = np.random.uniform(0.0, 0.6, n_star // 2)
        colors[n_star // 2:] = np.random.uniform(3.4, 4.0, n_star // 2)

        flux_1 = np.ones(n_star)
        mag_1 = (flux_1*units.nJy).to_value(units.ABmag)
        mag_2 = mag_1 - colors
        flux_2 = (mag_2*units.ABmag).to_value(units.nJy)

        flux_source = np.zeros(n_star) + 10000.0

        model_flux = spline.apply(flux_1, flux_2, flux_source)

        out_of_bounds = ((colors < 0.5) | (colors > 3.5))

        np.testing.assert_array_equal(np.isfinite(model_flux[out_of_bounds]), False)
        np.testing.assert_array_equal(np.isfinite(model_flux[~out_of_bounds]), True)

    def test_spline_serialize(self):
        spline = ColortermSpline(
            self._source_survey,
            self._target_survey,
            self._source_color_field_1,
            self._source_color_field_2,
            self._source_field,
            self._nodes,
            self._values,
            flux_offset=self._flux_offset,
        )

        self.assertEqual(spline.source_survey, self._source_survey)
        self.assertEqual(spline.target_survey, self._target_survey)
        self.assertEqual(spline.source_color_field_1, self._source_color_field_1)
        self.assertEqual(spline.source_color_field_2, self._source_color_field_2)
        self.assertEqual(spline.source_field, self._source_field)
        np.testing.assert_array_almost_equal(spline.nodes, self._nodes)
        np.testing.assert_array_almost_equal(spline.spline_values, self._values)
        self.assertEqual(spline.flux_offset, self._flux_offset)

        with tempfile.TemporaryDirectory() as temp_dir:
            filename = os.path.join(temp_dir, "test.yaml")

            spline.save(filename)

            spline2 = ColortermSpline.load(filename)

        self.assertEqual(spline2.source_survey, spline.source_survey)
        self.assertEqual(spline2.target_survey, spline.target_survey)
        self.assertEqual(spline2.source_color_field_1, spline.source_color_field_1)
        self.assertEqual(spline2.source_color_field_2, spline.source_color_field_2)
        self.assertEqual(spline2.source_field, spline.source_field)
        np.testing.assert_array_almost_equal(spline2.nodes, spline.nodes)
        np.testing.assert_array_almost_equal(spline2.spline_values, spline.spline_values)
        self.assertEqual(spline2.flux_offset, spline.flux_offset)

    def test_spline_overwrite(self):
        spline = ColortermSpline(
            self._source_survey,
            self._target_survey,
            self._source_color_field_1,
            self._source_color_field_2,
            self._source_field,
            self._nodes,
            self._values,
            flux_offset=self._flux_offset,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            filename = os.path.join(temp_dir, "test.yaml")

            spline.save(filename)

            with self.assertRaises(OSError):
                spline.save(filename)

            spline.save(filename, overwrite=True)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
