import os
import tempfile

import lsst.utils

from lsst.the.monster import MatchAndTransform
from test_catalog_measurement import TestGaiaDR3Info, TestGaiaXPInfo
import fitsio

ROOT = os.path.abspath(os.path.dirname(__file__))


class MonsterMatchAndTransformTest(lsst.utils.tests.TestCase):
    def setUp(self):
        self.GaiaDR3CatInfoClass = TestGaiaDR3Info
        self.TargetCatInfoClass = TestGaiaXPInfo
        self.TargetCatInfoClass.name = "GaiaXP"
        self.outputColumns = ('id',
                              'TestGaiaDR3_id',
                              'coord_ra',
                              'coord_dec',
                              'decam_g_from_GaiaXP_flux',
                              'decam_r_from_GaiaXP_flux',
                              'decam_i_from_GaiaXP_flux',
                              'decam_z_from_GaiaXP_flux',
                              'decam_y_from_GaiaXP_flux',
                              'decam_g_from_GaiaXP_fluxErr',
                              'decam_r_from_GaiaXP_fluxErr',
                              'decam_i_from_GaiaXP_fluxErr',
                              'decam_z_from_GaiaXP_fluxErr',
                              'decam_y_from_GaiaXP_fluxErr')

    def test_MatchAndTransform(self):
        """
        Test the MatchAndTransform function, by checking the
        length of the output and the output columns.

        Args:
            htmid (int): The HTM ID.
            catalog_list (list of str): The list of catalog names.
            write_path (str): The path to write the output file.
            gaia_cat_info_class: RefcatInfo for Gaia DR3 catalog.
        """

        # Set up the test.
        htmid = 147267

        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)
            mat = MatchAndTransform(catalog_info_list=[self.TargetCatInfoClass],
                                    write_path_inp=temp_dir,
                                    gaia_reference_class=self.GaiaDR3CatInfoClass,
                                    )

            # Run the MatchAndTransform function.
            mat.run(htmid=htmid)

            # Read the output file.
            output = fitsio.read(os.path.join(temp_dir, str(htmid) + ".fits"))
            print(output.dtype.names)
            # Check the output.
            self.assertEqual(output.shape, (347,))
            self.assertEqual(output.dtype.names, self.outputColumns)
