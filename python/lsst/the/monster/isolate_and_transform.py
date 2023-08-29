import numpy as np
import matplotlib.pyplot as plt
from smatch import Matcher
import warnings

from lsst.sphgeom import Box, HtmPixelization

from .splinecolorterms import ColortermSplineFitter, ColortermSpline
from .refcats import GaiaXPInfo, GaiaDR3Info, DESInfo, SkyMapperInfo, PS1Info, VSTInfo
from .utils import read_stars


"""
read in gaia shard
run isolated on gaia shard
read in each catalog
match with gaiadr3
for each band, interpolate DES magnitudes
Make sure interpolation behaves as expected (nan) outside range
save catalog shard


Question: current utils.read_stars takes a list of htm ids and reads in large catalogs 
do we pass 1 pixel at a time

could use this to get htmids
 ra_min, ra_max, dec_min, dec_max = self.ra_dec_range

        box = Box.fromDegrees(ra_min, dec_min, ra_max, dec_max)

        pixelization = HtmPixelization(7)
        rs = pixelization.envelope(box)
        
for filepaths and other config try to use: 
class GaiaXPSplineMeasurer(SplineMeasurer):
    CatInfoClass = GaiaXPInfo


"""

class MatchAndTransform:
    def run(htmid):
        
        # read in gaiaDR3 cat htmid
        gaia_stars=
        # isolate the gaia cat 
        gaia_stars=self._remove_neighbors(
                _cat
            )
        # loop over other catalogs
        for cat_info in catalog_list:
            # catalog_list should be a list of 
            #cat_info = self.CatInfoClass() e.g. gaia cat
            #read in star cat (if it exists)
            cat_stars=
            #match with gaia_stars
            with Matcher(gaia_stars["coord_ra"], gaia_stars["coord_dec"]) as m:
                idx, i1, i2, d = m.query_knn(
                    cat_stars["coord_ra"],
                    cat_stars["coord_dec"],
                    distance_upper_bound=0.5/3600.0,
                    return_indices=True,
                )
            cat_stars = cat_stars[i2]
            
            # read in spline
            spline = ColortermSpline.load(filename)
            # apply colorterms to transform to des mag
            
            for band in cat_info.bands:
                band_1, band_2 = cat_info.get_color_bands(band)
                model = colorterm.apply(
                        cat_stars[cat_info.get_flux_field(band_1)],
                        cat_stars[cat_info.get_flux_field(band_2)],
                        cat_start[cat_info.get_flux_field(band)],
                    )
            
            #save shard
    def _remove_neighbors(self, catalog):
        isaTask = IsolatedStarAssociationTask()
        isaTask.config.ra_column = "coord_ra"
        isaTask.config.dec_column = "coord_dec"
        return isaTask._remove_neighbors(catalog)
    
