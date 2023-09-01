import os
import unittest
import numpy as np
import astropy.units as units
from scipy.stats import median_abs_deviation
import tempfile

import lsst.utils

from lsst.the.monster import MatchAndTransform, 
from test_catalog_measurement import TestGaiaDR3Info, TestGaiaXPInfo


class MonsterMatchAndTransformTest(lsst.utils.tests.TestCase):
    def check_splinefitter(self, mag_offset=0.0, check_pars=True):
        """Check the spline fitter, with optional magnitude offset.

        Parameters
        ----------
        mag_offset : `float`, optional
            Overall magnitude offset between ref and meas catalogs.
        check_pars : `bool`, optional
            Check the parameter values? (Only works well if mag_offset=0.0).
        """
        def setUp(self):
        use test gaia_Dr3 data
        TestGaiaDR3Info, TestGaiaXPInfo
        
        
        check shape and columns in output
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)
            
            
            run(self, 
            htmid=None, 
            catalog_list=[GaiaXPInfo, SkyMapperInfo, PS1Info, VSTInfo],
            write_path=None
    ):
