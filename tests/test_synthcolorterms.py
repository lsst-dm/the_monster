import os
import unittest
import tempfile

# Ensure that matplotlib doesn't try to open a display during testing.
import matplotlib
matplotlib.use("Agg")

import lsst.utils  # noqa: E402

from lsst.the.monster import SynthLSSTSplineMeasurer, SynthLSSTInfo  # noqa: E402

ROOT = os.path.abspath(os.path.dirname(__file__))


class SynthLSSTInfoTester(SynthLSSTInfo):
    NAME = "SynthLSSTTester"
    bands = ["r"]


class SynthLSSTSplineMeasurerTester(SynthLSSTSplineMeasurer):
    THROUGHPUT_PATH = os.path.join(ROOT, "data", "throughputs")
    SynthLSSTInfoClass = SynthLSSTInfoTester


class SynthColortermSplineFitterTest(lsst.utils.tests.TestCase):
    def test_synth_des_lsst_colorterms(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)

            measurer = SynthLSSTSplineMeasurerTester()
            yaml_files = measurer.measure_synth_spline_fit(["r"])

            # One band.  Just check that it runs okay.
            self.assertEqual(len(yaml_files), 1)
            self.assertTrue(os.path.isfile(yaml_files[0]))
            cterm_file = "DES_to_SynthLSSTTester_band_r_color_term.png"
            self.assertTrue(os.path.isfile(cterm_file))


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
