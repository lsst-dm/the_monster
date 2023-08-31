import numpy as np
import matplotlib.pyplot as plt
import os
import fitsio
from smatch import Matcher
import warnings

from lsst.sphgeom import Box, HtmPixelization
from lsst.pipe.tasks.isolatedStarAssociation import IsolatedStarAssociationTask

from splinecolorterms import ColortermSplineFitter, ColortermSpline
from refcats import GaiaXPInfo, GaiaDR3Info, DESInfo, SkyMapperInfo, PS1Info, VSTInfo
from utils import read_stars

__all__ = ["MatchAndTransform"]

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
    """Match catalogs to Gaia and transform them to 'the_monster' reference frame.

    Parameters
    ----------
    htmid : `int`
        htm level 7 id of catalog(s).
    """
    GaiaDR3CatInfoClass = GaiaDR3Info
    testing_mode = False

    def run(self, htmid=None, catalog_list=[GaiaXPInfo, SkyMapperInfo, PS1Info, VSTInfo]):
        
        # read in gaiaDR3 cat htmid
        # Read in the Gaia stars in the htmid.
        gaia_info = self.GaiaDR3CatInfoClass()

        gaia_stars_all = read_stars(gaia_info.path, [htmid], allow_missing=self.testing_mode)

        # isolate the gaia cat 
        gaia_stars = self._remove_neighbors(
                gaia_stars_all
            )
        # loop over other catalogs
        for cat_info in catalog_list:
            # Create an output directory name for the shards:
            cat_outdir = cat_info().name+'_transformed'

            # catalog_list should be a list of 
            #cat_info = self.CatInfoClass() e.g. gaia cat
            #read in star cat (if it exists)
            if os.path.isfile(cat_info().path+'/'+str(htmid)+'.fits'):
                cat_stars = read_stars(cat_info().path, [htmid], allow_missing=self.testing_mode)

                #match with gaia_stars
                with Matcher(gaia_stars["coord_ra"], gaia_stars["coord_dec"]) as m:
                    idx, i1, i2, d = m.query_knn(
                        cat_stars["coord_ra"],
                        cat_stars["coord_dec"],
                        distance_upper_bound=0.5/3600.0,
                        return_indices=True,
                    )
                cat_stars = cat_stars[i2]

                for band in cat_info.bands:
                    # yaml spline fits are per-band, so loop over bands
                    # read in spline
                    filename = '../../../../colorterms/'+cat_info().name+'_to_DES_band_'+str(band)+'.yaml'
                    # Fix the path above to be relative to the_monster package root

                    colorterm_spline = ColortermSpline.load(filename)

                    # apply colorterms to transform to des mag
                    band_1, band_2 = cat_info().get_color_bands(band)
                    model = colorterm_spline.apply(
                            cat_stars[cat_info().get_flux_field(band_1)],
                            cat_stars[cat_info().get_flux_field(band_2)],
                            cat_stars[cat_info().get_flux_field(band)],
                        )
                    # Append the modeled mags column to cat_stars
                    cat_stars.add_column(model, name=cat_info().name+'_'+model.name)
                
                write_path = 'tmp/'+cat_outdir
                # write_path = cat_info().path+'/'+cat_outdir+'/'
                if os.path.exists(write_path)==False:
                    os.makedirs(write_path)
                write_path += f"/{htmid}.fits"
                # Save the shard to FITS. Should probably use fitsio instead of Table.write?
                cat_stars.write(write_path, overwrite=True)

            else:
                print(cat_info().path+'/'+str(htmid)+'.fits does not exist.')

        # import pdb; pdb.set_trace()

    def _remove_neighbors(self, catalog):
        isaTask = IsolatedStarAssociationTask()
        isaTask.config.ra_column = "coord_ra"
        isaTask.config.dec_column = "coord_dec"
        return isaTask._remove_neighbors(catalog)
