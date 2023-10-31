import os
import unittest
import tempfile

# Ensure that matplotlib doesn't try to open a display during testing.
import matplotlib
matplotlib.use("Agg")

import lsst.utils  # noqa: E402

from lsst.the.monster import GaiaDR3Info, GaiaXPInfo, DESInfo, PS1Info  # noqa: E402
from lsst.the.monster import GaiaXPSplineMeasurer, PS1SplineMeasurer  # noqa: E402


ROOT = os.path.abspath(os.path.dirname(__file__))


class GaiaDR3InfoTester(GaiaDR3Info):
    PATH = os.path.join(ROOT, "data", "gaia_dr3")
    NAME = "TestGaiaDR3"


class GaiaXPInfoTester(GaiaXPInfo):
    PATH = os.path.join(ROOT, "data", "gaia_xp")
    NAME = "TestGaiaXP"
    COLORTERM_PATH = os.path.join(ROOT, "data", "colorterms")


class DESInfoTester(DESInfo):
    PATH = os.path.join(ROOT, "data", "des")
    NAME = "TestDES"


class PS1InfoTester(PS1Info):
    PATH = os.path.join(ROOT, "data", "ps1")
    NAME = "TestPS1"
    COLORTERM_PATH = os.path.join(ROOT, "data", "colorterms")


class GaiaXPSplineMeasurerTester(GaiaXPSplineMeasurer):
    CatInfoClass = GaiaXPInfoTester
    TargetCatInfoClass = DESInfoTester
    GaiaCatInfoClass = GaiaDR3InfoTester

    testing_mode = True

    @property
    def n_nodes(self):
        return 5


class PS1SplineMeasurerTester(PS1SplineMeasurer):
    CatInfoClass = PS1InfoTester
    TargetCatInfoClass = DESInfoTester
    GaiaCatInfoClass = GaiaDR3InfoTester

    MagOffsetCatInfoClass = GaiaXPInfoTester

    testing_mode = True

    @property
    def n_mag_nodes(self):
        return 5


class SplineMeasurerTest(lsst.utils.tests.TestCase):
    def test_measure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)

            measurer = GaiaXPSplineMeasurerTester()

            yaml_files = measurer.measure_spline_fit()

            # Check that the yaml files were created.
            for yaml_file in yaml_files:
                self.assertTrue(os.path.isfile(yaml_file))

            # And check for the QA plots.
            for band in ["g", "r", "i", "z", "y"]:
                self.assertTrue(os.path.isfile(f"TestGaiaXP_to_TestDES_band_{band}_color_term.png"))
                self.assertTrue(os.path.isfile(f"TestGaiaXP_to_TestDES_band_{band}_flux_residuals.png"))

    def test_ps1_measure(self):
        # This is a separate test because of teh secondary matching.
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)

            measurer = PS1SplineMeasurerTester()

            yaml_files = measurer.measure_spline_fit()

            # Check that the yaml files were created.
            for yaml_file in yaml_files:
                self.assertTrue(os.path.isfile(yaml_file))

            # And check for the QA plots.
            for band in ["g", "r", "i", "z", "y"]:
                self.assertTrue(os.path.isfile(f"TestPS1_to_TestDES_band_{band}_color_term.png"))
                self.assertTrue(os.path.isfile(f"TestPS1_to_TestDES_band_{band}_flux_residuals.png"))
                # And another QA plot
                self.assertTrue(os.path.isfile(f"TestPS1_vs_TestGaiaXP_band_{band}_mag_offset.png"))


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
