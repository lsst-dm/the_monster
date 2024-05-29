import os
import unittest
import tempfile

# Ensure that matplotlib doesn't try to open a display during testing.
import matplotlib
matplotlib.use("Agg")

import lsst.utils  # noqa: E402

from lsst.the.monster import GaiaDR3Info, GaiaXPInfo, GaiaXPuInfo, DESInfo, PS1Info, SDSSInfo  # noqa: E402
from lsst.the.monster import UBandOffsetMapMaker  # noqa: E402


ROOT = os.path.abspath(os.path.dirname(__file__))


class GaiaDR3InfoTester(GaiaDR3Info):
    PATH = os.path.join(ROOT, "data", "gaia_dr3")
    NAME = "TestGaiaDR3"


class GaiaXPInfoTester(GaiaXPInfo):
    PATH = os.path.join(ROOT, "data", "gaia_xp")
    NAME = "TestGaiaXP"
    COLORTERM_PATH = os.path.join(ROOT, "data", "colorterms")


class GaiaXPuInfoTester(GaiaXPuInfo):
    PATH = os.path.join(ROOT, "data", "gaia_xp")
    ORIG_NAME_FOR_TEST = "GaiaXPu"
    NAME = "TestGaiaXPu"
    COLORTERM_PATH = os.path.join(ROOT, "data", "colorterms")


class DESInfoTester(DESInfo):
    PATH = os.path.join(ROOT, "data", "des")
    ORIG_NAME_FOR_TEST = "DES"
    NAME = "TestDES"
    COLORTERM_PATH = os.path.join(ROOT, "data", "colorterms")


class PS1InfoTester(PS1Info):
    PATH = os.path.join(ROOT, "data", "ps1")
    TRANSFORMED_PATH = os.path.join(ROOT, "data", "ps1_transformed")
    ORIG_NAME_FOR_TEST = "PS1"
    NAME = "TestPS1"
    COLORTERM_PATH = os.path.join(ROOT, "data", "colorterms")


class SDSSInfoTester(SDSSInfo):
    PATH = os.path.join(ROOT, "data", "sdss")
    NAME = "TestSDSS"
    COLORTERM_PATH = os.path.join(ROOT, "data", "colorterms")


class UBandOffsetMapMakerTest(lsst.utils.tests.TestCase):
    def test_measure_uband_offset_map(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)

            measurer = UBandOffsetMapMaker(
                gaia_reference_class=GaiaDR3InfoTester,
                catalog_info_class_list=[PS1InfoTester, DESInfoTester],
                uband_ref_class=GaiaXPuInfoTester,
                uband_slr_class=DESInfoTester,
                testing_mode=True,
            )
            fname = measurer.measure_uband_offset_map()
            self.assertTrue(os.path.isfile(fname))

            measurer.plot_uband_offset_maps(fname, "uslr-uxp")

            self.assertTrue(os.path.isfile("uslr-uxp_full_map.png"))
            self.assertTrue(os.path.isfile("uslr-uxp_highglat_map.png"))
            self.assertTrue(os.path.isfile("uslr-uxp_nstar.png"))
            # The histogram fails in the tiny test.

    def test_measure_uband_offset_map_sdss(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)

            measurer = UBandOffsetMapMaker(
                gaia_reference_class=GaiaDR3InfoTester,
                catalog_info_class_list=[PS1InfoTester, DESInfoTester],
                uband_ref_class=SDSSInfoTester,
                testing_mode=True,
            )
            fname = measurer.measure_uband_offset_map()
            self.assertTrue(os.path.isfile(fname))

            measurer.plot_uband_offset_maps(fname, "uslr-usdss", mode="slr-sdss")

            self.assertTrue(os.path.isfile("uslr-usdss_full_map.png"))
            self.assertTrue(os.path.isfile("uslr-usdss_highglat_map.png"))
            self.assertTrue(os.path.isfile("uslr-usdss_nstar.png"))
            # The histogram fails in the tiny test.

    def test_measure_uband_direct_offset_map(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)

            measurer = UBandOffsetMapMaker(
                gaia_reference_class=GaiaDR3InfoTester,
                catalog_info_class_list=[GaiaXPuInfoTester],
                uband_ref_class=SDSSInfoTester,
                testing_mode=True,
            )
            fname = measurer.measure_uband_offset_map_direct()
            self.assertTrue(os.path.isfile(fname))

            measurer.plot_uband_offset_maps(fname, "uxp-usdss", mode="direct")

            self.assertTrue(os.path.isfile("uxp-usdss_full_map.png"))
            self.assertTrue(os.path.isfile("uxp-usdss_highglat_map.png"))
            self.assertTrue(os.path.isfile("uxp-usdss_nstar.png"))
            # The histogram fails in the tiny test.


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
