import esutil
import os
from smatch import Matcher
import numpy as np

from lsst.pipe.tasks.isolatedStarAssociation import IsolatedStarAssociationTask
from .splinecolorterms import ColortermSpline
from .refcats import GaiaXPInfo, GaiaDR3Info, SkyMapperInfo, PS1Info, VSTInfo, DESInfo
from .utils import read_stars, makeRefSchema, makeRefCat

__all__ = ["AssembleMonsterRefcat"]

"""
This Python script starts with the full Gaia DR3 catalog, then assembles a
monster refcat by doing the following:

For each shard:
1. Read the full Gaia DR3 catalog for the shard
2. Read each of the (already transformed to the DES system) refcats for the shard
3. Transform each refcat to the (synthetic) LSST system
4. Initialize 18 columns (ugrizy fluxes and their errors, and a source flag
    (integer)) with NaN values
5. Match each refcat to the Gaia DR3 catalog
6. Loop over surveys from lowest to highest priority, updating the fluxes, flux
    errors, and flags whenever a value is non-NaN

What we want for refcat:
* Start w/ (all of) Gaia DR3, keeping all those columns
* Transform from DES to LSST system (naming: “monster_lsst_{band}_flux” + error + source_flag)
* Add ugrizy flux, flux errors, and source flag (stored as integer) - 18 columns total. Set ones with no matches in any photom cats to NaNs (initialize them that way).
* Keep the whole sky (but don't bother vetting the northern sky)
* Remember what's in the transformed-to-DES catalogs have already been vetted, so no additional cuts need to happen.
* Can loop over surveys from lowest-to-highest priority, overwriting whenever there's a match (so that DES (highest priority) will be last)
* NOTE: color-terms for synth_LSST are over a narrower band than empirical-DES system. Check the validity before accepting.
"""


class AssembleMonsterRefcat:
    def __init__(self,
                 gaia_reference_class=GaiaDR3Info,
                 catalog_info_class_list=[VSTInfo, SkyMapperInfo,
                                          PS1Info, GaiaXPInfo, DESInfo],
                 write_path_inp=None,
                 testing_mode=False,
                 ):

        self.gaia_reference_info = gaia_reference_class()
        self.catalog_info_class_list = [cat_info() for cat_info
                                        in catalog_info_class_list]
        self.testing_mode = testing_mode
        self.write_path_inp = write_path_inp
        self.all_bands = ['u', 'g', 'r', 'i', 'z', 'y']

    def run(self,
            *,
            htmid,
            verbose=False,
            ):
        """Match catalogs to Gaia and transform them to 'the_monster'
           reference frame.

        Parameters
        ----------
        htmid : `int`
            HTM id of the catalogs.
        """

        # Read in the Gaia stars in the htmid.
        gaia_stars_all = read_stars(self.gaia_reference_info.path, [htmid],
                                    allow_missing=self.testing_mode)
        
        # Initialize output columns for the fluxes, flux errors, and flags,
        # with all of them set to NaN by default:
        nan_column = np.full(len(gaia_stars_all["id"]), np.nan)

        for band in self.all_bands:
            gaia_stars_all.add_column(nan_column,
                                      name=f"monster_lsst_{band}_flux")
            gaia_stars_all.add_column(nan_column,
                                      name=f"monster_lsst_{band}_fluxErr")
            gaia_stars_all.add_column(nan_column,
                                      name=f"monster_lsst_{band}_source_flag")
        
            # Loop over the refcats
            for cat_info in self.catalog_info_class_list:
                # catalog_info_class_list should be a list of
                # cat_info = self.CatInfoClass() e.g. gaia cat

                # Read in star cat that has already been transformed to the DES
                # system (if it exists). Note the use of "write_path" instead
                # of "path" here, so that it gets the transformed catalog.
                if os.path.isfile(cat_info.write_path+'/'+str(htmid)+'.fits'):
                    cat_stars = read_stars(cat_info.write_path, [htmid],
                                           allow_missing=self.testing_mode)
                    
                    # Transform from the DES to the synthetic LSST system:
                    # I need to figure out how to get the synthetic colorterm files
                    # and apply them. 

                    """
                    # read in spline
                    filename = cat_info.colorterm_file(band)
                    colorterm_spline = ColortermSpline.load(filename)

                    # apply colorterms to transform to des mag
                    band_1, band_2 = cat_info.get_color_bands(band)
                    orig_flux = cat_stars[cat_info.get_flux_field(band)]
                    orig_flux_err = cat_stars[cat_info.get_flux_field(band)+'Err']
                    model_flux = colorterm_spline.apply(
                        cat_stars[cat_info.get_flux_field(band_1)],
                        cat_stars[cat_info.get_flux_field(band_2)],
                        orig_flux,
                    )

                    # Rescale flux error to keep S/N constant
                    model_flux_err = model_flux * (orig_flux_err/orig_flux)

                    # Append the modeled flux columns to cat_stars
                    cat_stars.add_column(model_flux,
                                         name=f"decam_{band}_from_{cat_info.name}_flux")
                    cat_stars.add_column(model_flux_err,
                                         name=f"decam_{band}_from_{cat_info.name}_fluxErr")
                    """

                    # Match the output catalog to Gaia.
                    a, b = esutil.numpy_util.match(gaia_stars_all['id'], output['id'])

