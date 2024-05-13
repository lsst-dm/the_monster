import unittest
import esutil
import numpy as np
import os
import tempfile

import lsst.afw as afw
import lsst.utils

from lsst.the.monster import AssembleMonsterRefcat
from lsst.the.monster.utils import read_stars
from test_catalog_measurement import GaiaDR3InfoTester, GaiaXPInfoTester

ROOT = os.path.abspath(os.path.dirname(__file__))


class MonsterAssembleTest(lsst.utils.tests.TestCase):
    def setUp(self):
        self.GaiaDR3CatInfoClass = GaiaDR3InfoTester
        self.TargetCatInfoClass = GaiaXPInfoTester
        self.synthSystem = 'LSST'
        self.outputColumns = ['id',
                              'coord_ra',
                              'coord_dec',
                              'phot_g_mean_flux',
                              'phot_bp_mean_flux',
                              'phot_rp_mean_flux',
                              'phot_g_mean_fluxErr',
                              'phot_bp_mean_fluxErr',
                              'phot_rp_mean_fluxErr',
                              'coord_raErr',
                              'coord_decErr',
                              'epoch',
                              'pm_ra',
                              'pm_dec',
                              'pm_raErr',
                              'pm_decErr',
                              'pm_flag',
                              'parallax',
                              'parallaxErr',
                              'parallax_flag',
                              'coord_ra_coord_dec_Cov',
                              'coord_ra_pm_ra_Cov',
                              'coord_ra_pm_dec_Cov',
                              'coord_ra_parallax_Cov',
                              'coord_dec_pm_ra_Cov',
                              'coord_dec_pm_dec_Cov',
                              'coord_dec_parallax_Cov',
                              'pm_ra_pm_dec_Cov',
                              'pm_ra_parallax_Cov',
                              'pm_dec_parallax_Cov',
                              'astrometric_excess_noise',
                              'monster_'+str.lower(self.synthSystem)+'_u_flux',
                              'monster_'+str.lower(self.synthSystem)+'_u_fluxErr',
                              'monster_'+str.lower(self.synthSystem)+'_u_source_flag',
                              'monster_'+str.lower(self.synthSystem)+'_g_flux',
                              'monster_'+str.lower(self.synthSystem)+'_g_fluxErr',
                              'monster_'+str.lower(self.synthSystem)+'_g_source_flag',
                              'monster_'+str.lower(self.synthSystem)+'_r_flux',
                              'monster_'+str.lower(self.synthSystem)+'_r_fluxErr',
                              'monster_'+str.lower(self.synthSystem)+'_r_source_flag',
                              'monster_'+str.lower(self.synthSystem)+'_i_flux',
                              'monster_'+str.lower(self.synthSystem)+'_i_fluxErr',
                              'monster_'+str.lower(self.synthSystem)+'_i_source_flag',
                              'monster_'+str.lower(self.synthSystem)+'_z_flux',
                              'monster_'+str.lower(self.synthSystem)+'_z_fluxErr',
                              'monster_'+str.lower(self.synthSystem)+'_z_source_flag',
                              'monster_'+str.lower(self.synthSystem)+'_y_flux',
                              'monster_'+str.lower(self.synthSystem)+'_y_fluxErr',
                              'monster_'+str.lower(self.synthSystem)+'_y_source_flag']

    def test_AssembleMonsterRefCat(self):
        """
        Test the AssembleMonsterRefcat function, by checking the
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
            amr = AssembleMonsterRefcat(catalog_info_class_list=[self.TargetCatInfoClass],
                                        monster_path_inp=temp_dir,
                                        gaia_reference_class=self.GaiaDR3CatInfoClass,
                                        synth_system=self.synthSystem
                                        )

            # Run the AssembleMonsterRefcat function.
            amr.run(htmid=htmid)

            # Read the output file.
            fits_path = os.path.join(temp_dir, str(htmid) + ".fits")
            output = afw.table.SimpleCatalog.readFits(fits_path)

            # Check the output.
            self.assertEqual(len(output), 1579)
            self.assertEqual(output.schema.getOrderedNames(), self.outputColumns)

            # Check that the positions are the same for this catalog and Gaia
            # Read in the Gaia stars in the htmid.
            gaia_stars_all = read_stars(self.GaiaDR3CatInfoClass().path, [htmid],
                                        allow_missing=False)
            # Match the output catalog to Gaia.
            a, b = esutil.numpy_util.match(gaia_stars_all['id'], output['id'])

            # Check that the coordinates are the same.
            np.testing.assert_array_almost_equal(gaia_stars_all[a]['coord_ra'],
                                                 np.rad2deg(output['coord_ra'][b]))
            np.testing.assert_array_almost_equal(gaia_stars_all[a]['coord_dec'],
                                                 np.rad2deg(output['coord_dec'][b]))


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
