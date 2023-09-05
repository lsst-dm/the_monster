import os
import unittest
import numpy as np
import astropy.units as units
from scipy.stats import median_abs_deviation
import tempfile

import lsst.utils

from lsst.the.monster import MatchAndTransform 
from test_catalog_measurement import TestGaiaDR3Info, TestGaiaXPInfo
import fitsio


class MonsterMatchAndTransformTest(lsst.utils.tests.TestCase):
    def setUp(self):
            self.GaiaDR3CatInfoClass = TestGaiaDR3Info
            self.TargetCatInfoClass = TestGaiaXPInfo
            self.TargetCatInfoClass.name = "GaiaXP"
            self.outputColumns= ('id',
                                'coord_ra',
                                'coord_dec',
                                'decam_g_flux_from_GaiaXP',
                                'decam_r_flux_from_GaiaXP',
                                'decam_i_flux_from_GaiaXP',
                                'decam_z_flux_from_GaiaXP',
                                'decam_y_flux_from_GaiaXP')

    def test_MatchAndTransform(self):
        """
        Test the MatchAndTransform function, by checking the 
        length of the output and the output columns. 

        Args:
            htmid (int): The HTM ID.
            catalog_list (list of str): The list of catalog names.
            write_path (str): The path to write the output file.
        """

        # Set up the test.
        htmid = 147267
        MAT = MatchAndTransform()
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)

            # Run the MatchAndTransform function.
            MAT.run(htmid=htmid,
                    catalog_list=[self.TargetCatInfoClass],
                    write_path=temp_dir
                    )

            # Read the output file.
            output = fitsio.read(os.path.join(temp_dir, str(htmid) + ".fits"))

            # Check the output.
            self.assertEqual(output.shape, (340,))
            self.assertEqual(output.dtype.names, self.outputColumns)

