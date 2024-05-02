import os
import unittest
import tempfile

# Ensure that matplotlib doesn't try to open a display during testing.
import matplotlib
matplotlib.use("Agg")

import lsst.utils  # noqa: E402

from lsst.the.monster import GaiaDR3Info, GaiaXPuInfo, DESInfo, PS1Info  # noqa: E402
from lsst.the.monster import UBandSLRSplineMeasurer  # noqa: E402


ROOT = os.path.abspath(os.path.dirname(__file__))


class GaiaDR3InfoTester(GaiaDR3Info):
    PATH = os.path.join(ROOT, "data", "gaia_dr3")
    NAME = "TestGaiaDR3"


class GaiaXPuInfoTester(GaiaXPuInfo):
    PATH = os.path.join(ROOT, "data", "gaia_xp")
    NAME = "TestGaiaXPu"
    COLORTERM_PATH = os.path.join(ROOT, "data", "colorterms")


class DESInfoTester(DESInfo):
    WRITE_PATH = os.path.join(ROOT, "data", "des_transformed")
    PATH = os.path.join(ROOT, "data", "des")
    ORIG_NAME_FOR_TEST = "DES"
    NAME = "TestDES"


class PS1InfoTester(PS1Info):
    WRITE_PATH = os.path.join(ROOT, "data", "ps1_transformed")
    PATH = os.path.join(ROOT, "data", "ps1")
    ORIG_NAME_FOR_TEST = "PS1"
    NAME = "TestPS1"
    COLORTERM_PATH = os.path.join(ROOT, "data", "colorterms")


class UBandSLRSplineMeasurerTester(UBandSLRSplineMeasurer):
    @property
    def ra_dec_range(self):
        return (40.0, 60.0, -30.0, -20.0)

    @property
    def n_nodes(self):
        return 5

    @property
    def n_mag_nodes(self):
        return 5


class UBandSLRSplineMeasurerTest(lsst.utils.tests.TestCase):
    def test_measure_uband_slr_spline(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)

            measurer = UBandSLRSplineMeasurerTester(
                gaia_reference_class=GaiaDR3InfoTester,
                catalog_info_class_list=[
                    PS1InfoTester,
                    DESInfoTester,
                ],
                uband_ref_class=GaiaXPuInfoTester,
                uband_slr_class=DESInfoTester,
                testing_mode=True,
            )

            yaml_files = measurer.measure_uband_slr_spline_fit()

            for yaml_file in yaml_files:
                self.assertTrue(os.path.isfile(yaml_file))

            self.assertTrue(os.path.isfile("transformed_to_TestGaiaXPu_band_g_slr.png"))
            self.assertTrue(os.path.isfile("transformed_to_TestGaiaXPu_band_g_slr_flux_residuals.png"))
            self.assertTrue(os.path.isfile("transformed_to_TestGaiaXPu_band_g_slr_mag_offset.png"))


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
