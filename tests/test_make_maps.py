import unittest
import os
import tempfile

# Ensure that matplotlib doesn't try to open a display during testing.
import matplotlib
matplotlib.use("Agg")

import lsst.utils  # noqa: E402


from lsst.the.monster import PS1MinusGaiaXPOffsetMapMaker  # noqa: E402
from lsst.the.monster import MatchAndTransform  # noqa: E402
from test_catalog_measurement import GaiaDR3InfoTester, GaiaXPInfoTester, PS1InfoTester  # noqa: E402

ROOT = os.path.abspath(os.path.dirname(__file__))


class MonsterMakeOffsetMapsTest(lsst.utils.tests.TestCase):
    def test_makeOffsetMapAndPlot(self):
        # Set up the test.
        htmid = 147267

        gaia_xp_info = GaiaXPInfoTester()
        ps1_info = PS1InfoTester()

        with tempfile.TemporaryDirectory() as temp_dir:
            # Use yet another way of hacking in test paths
            PS1InfoTester.TRANSFORMED_PATH = os.path.join(temp_dir, f"{gaia_xp_info.name}_transformed")
            GaiaXPInfoTester.TRANSFORMED_PATH = os.path.join(temp_dir, f"{ps1_info.name}_transformed")

            class PS1MinusGaiaXPOffsetMapMakerTester(PS1MinusGaiaXPOffsetMapMaker):
                MinuendInfoClass = PS1InfoTester
                SubtrahendInfoClass = GaiaXPInfoTester

            os.chdir(temp_dir)

            # We first need to do the match and transform.
            mat = MatchAndTransform(
                catalog_info_class_list=[GaiaXPInfoTester, PS1InfoTester],
                gaia_reference_class=GaiaDR3InfoTester,
            )
            mat.run(htmid=htmid)

            maker = PS1MinusGaiaXPOffsetMapMakerTester()
            hsp_file = maker.measure_offset_map()

            self.assertTrue(os.path.isfile(hsp_file))

            maker.plot_offset_maps(hsp_file)

            for band in ["g", "r", "i", "z", "y"]:
                self.assertTrue(os.path.isfile(f"TestPS1-TestGaiaXP_fullmap_{band}.png"))
                self.assertTrue(os.path.isfile(f"TestPS1-TestGaiaXP_highglat_{band}.png"))
                # This one fails the fit, which is fine, it's a terrible fit.
                if band != "z":
                    self.assertTrue(os.path.isfile(f"TestPS1-TestGaiaXP_highglat_hist_{band}.png"))


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
