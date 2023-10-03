import esutil
import numpy as np
import os
import tempfile

import lsst.afw as afw
import lsst.utils

from lsst.the.monster import MatchAndTransform
from lsst.the.monster.utils import read_stars
from test_catalog_measurement import TestGaiaDR3Info, TestGaiaXPInfo

ROOT = os.path.abspath(os.path.dirname(__file__))


class MonsterMatchAndTransformTest(lsst.utils.tests.TestCase):
    def setUp(self):
        self.GaiaDR3CatInfoClass = TestGaiaDR3Info
        self.TargetCatInfoClass = TestGaiaXPInfo
        self.TargetCatInfoClass.name = "GaiaXP"
        self.outputColumns = ['id',
                              'coord_ra',
                              'coord_dec',
                              'TestGaiaDR3_id',
                              'decam_g_from_GaiaXP_flux',
                              'decam_g_from_GaiaXP_fluxErr',
                              'decam_r_from_GaiaXP_flux',
                              'decam_r_from_GaiaXP_fluxErr',
                              'decam_i_from_GaiaXP_flux',
                              'decam_i_from_GaiaXP_fluxErr',
                              'decam_z_from_GaiaXP_flux',
                              'decam_z_from_GaiaXP_fluxErr',
                              'decam_y_from_GaiaXP_flux',
                              'decam_y_from_GaiaXP_fluxErr']

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
            mat = MatchAndTransform(catalog_info_class_list=[self.TargetCatInfoClass],
                                    write_path_inp=temp_dir,
                                    gaia_reference_class=self.GaiaDR3CatInfoClass,
                                    )

            # Run the MatchAndTransform function.
            mat.run(htmid=htmid)

            # Read the output file.
            fits_path = os.path.join(temp_dir, str(htmid) + ".fits")
            output = afw.table.SimpleCatalog.readFits(fits_path)

            # Check the output.
            self.assertEqual(len(output), 347)
            self.assertEqual(output.schema.getOrderedNames(), self.outputColumns)

            # Check that the positions are the same for this catalog and Gaia
            # Read in the Gaia stars in the htmid.
            gaia_stars_all = read_stars(self.GaiaDR3CatInfoClass().path, [htmid],
                                        allow_missing=False)
            # Match the output catalog to Gaia.
            a, b = esutil.numpy_util.match(gaia_stars_all['id'], output['id'])

            # Check that the coordinates are the same (note that the output
            # catalog is in radians, so convert to degrees first).
            # import pytest; pytest.set_trace()
            np.testing.assert_array_almost_equal(gaia_stars_all[a]['coord_ra'],
                                                 np.rad2deg(output['coord_ra'][b]))
            np.testing.assert_array_almost_equal(gaia_stars_all[a]['coord_dec'],
                                                 np.rad2deg(output['coord_dec'][b]))
