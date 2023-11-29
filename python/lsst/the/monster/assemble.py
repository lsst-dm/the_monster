import esutil
import os
import numpy as np
import lsst.utils

from .splinecolorterms import ColortermSpline
from .refcats import GaiaXPInfo, GaiaDR3Info, SkyMapperInfo, PS1Info, VSTInfo, DESInfo, SynthLSSTInfo
from .utils import read_stars, makeMonsterSchema, makeMonsterCat

__all__ = ["AssembleMonsterRefcat"]

"""
This Python script starts with the full Gaia DR3 catalog, then assembles a
monster refcat by doing the following:

For each shard:
1. Read the full Gaia DR3 catalog for the shard
2. Read each of the (already transformed to the DES system) refcats for the
    shard
3. Transform each refcat to the (synthetic) LSST system
4. Initialize 18 columns (ugrizy fluxes and their errors, and a source flag
    (integer)) with NaN values
5. Match each refcat to the Gaia DR3 catalog
6. Loop over surveys from lowest to highest priority, updating the fluxes, flux
    errors, and flags whenever a value is non-NaN

What we want for refcat:
* Start w/ (all of) Gaia DR3, keeping all those columns
* Transform from DES to LSST system (naming:
    “monster_lsst_{band}_flux” + error + source_flag)
* Add ugrizy flux, flux errors, and source flag (stored as integer) -
    18 columns total. Set ones with no matches in any photom cats to NaNs
    (initialize them that way).
* Keep the whole sky (but don't bother vetting the northern sky)
* Remember what's in the transformed-to-DES catalogs have already been vetted,
    so no additional cuts need to happen.
* Can loop over surveys from lowest-to-highest priority, overwriting whenever
    there's a match (so that DES (highest priority) will be last)
* NOTE: color-terms for synth_LSST are over a narrower band than empirical-DES
    system. Check the validity before accepting.
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
        self.synth_lsst_info = SynthLSSTInfo()
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

        # Initialize output columns for the fluxes and flux errors,
        # with all of them set to NaN by default. Also initialize flag columns
        # with "-1" values as default.
        nan_column = np.full(len(gaia_stars_all["id"]), np.nan)
        int_column = np.full(len(gaia_stars_all["id"]), -1)

        for band in self.all_bands:
            gaia_stars_all.add_column(nan_column,
                                      name=f"monster_lsst_{band}_flux")
            gaia_stars_all.add_column(nan_column,
                                      name=f"monster_lsst_{band}_fluxErr")
            gaia_stars_all.add_column(int_column,
                                      name=f"monster_lsst_{band}_source_flag")

            # Loop over the refcats
            for cat_info in self.catalog_info_class_list:
                # catalog_info_class_list should be a list of
                # cat_info = self.CatInfoClass() e.g. gaia cat

                # Read in star cat that has already been transformed to the DES
                # system (if it exists). Note the use of "write_path" instead
                # of "path" here, so that it gets the transformed catalog.
                if os.path.isfile(cat_info.write_path+'/'+str(htmid)+'.fits')\
                        and band in cat_info.bands:

                    cat_stars = read_stars(cat_info.write_path, [htmid],
                                           allow_missing=self.testing_mode)

                    # Transform from the DES to the synthetic LSST system:
                    colorterm_path = os.path.join(
                        lsst.utils.getPackageDir("the_monster"),
                        "colorterms",
                    )
                    colorterm_file_string = 'DES_to_SynthLSST_band'
                    colorterm_filename = os.path.join(
                        colorterm_path,
                        colorterm_file_string+f'_{band}.yaml',
                    )

                    # read in spline
                    colorterm_spline = ColortermSpline.load(colorterm_filename)

                    # apply colorterms to transform to SynthLSST mag
                    band_1, band_2 = cat_info.get_color_bands(band)
                    orig_flux = cat_stars[f"decam_{band}_from_{cat_info.name}_flux"]
                    orig_flux_err = cat_stars[f"decam_{band}_from_{cat_info.name}_fluxErr"]
                    model_flux = colorterm_spline.apply(
                        cat_stars[f"decam_{band_1}_from_{cat_info.name}_flux"],
                        cat_stars[f"decam_{band_2}_from_{cat_info.name}_flux"],
                        orig_flux,
                    )

                    # Rescale flux error to keep S/N constant
                    model_flux_err = model_flux * (orig_flux_err/orig_flux)

                    # Add the fluxes and their errors to the catalog:
                    cat_stars[f"monster_lsst_{band}_flux"] = model_flux
                    cat_stars[f"monster_lsst_{band}_fluxErr"] = model_flux_err

                    # Apply selection to only apply transformations within the
                    # useful color range. (Note that the input DES-system
                    # catalogs already had their survey-specific cuts applied,
                    # so these cuts should be for the SynthLSST
                    # transformations.)
                    color_range = self.synth_lsst_info.get_color_range(band)
                    colors = cat_info.get_transformed_mag_colors(cat_stars, band)
                    selected = (colors >= color_range[0]) & (colors <= color_range[1])

                    # selected = cat_info.select_stars(cat_stars, band)
                    flux_not_nan = np.isfinite(cat_stars[f"monster_lsst_{band}_flux"])
                    flag = selected & flux_not_nan
                    cat_stars_selected = cat_stars[flag]

                    # Match the transformed catalog to Gaia.
                    a, b = esutil.numpy_util.match(gaia_stars_all['id'],
                                                   cat_stars_selected['GaiaDR3_id'])

                    # If the flux measurement is OK, write it to the overall
                    # Gaia catalog:
                    flux_col = f"monster_lsst_{band}_flux"
                    gaia_stars_all[flux_col][a] = cat_stars_selected[flux_col][b]
                    fluxerr_col = flux_col+'Err'
                    gaia_stars_all[fluxerr_col][a] = cat_stars_selected[fluxerr_col][b]

                    # Update the flags to denote which survey the flux came
                    # from:
                    gaia_stars_all[f"monster_lsst_{band}_source_flag"][a] = cat_info.flag

        # Call this version "monster_v1":
        write_path_monster = "/sdf/data/rubin/shared/the_monster/sharded_refcats/monster_v1"

        # Output the finished catalog for the shard:
        if os.path.exists(write_path_monster) is False:
            os.makedirs(write_path_monster)
        write_path_monster += f"/{htmid}.fits"

        # Convert the refcat to a SimpleCatalog
        monsterSchema = makeMonsterSchema(gaia_stars_all.itercols(), self.all_bands)
        monsterCat = makeMonsterCat(monsterSchema, gaia_stars_all)

        # Save the shard to FITS.
        monsterCat.writeFits(write_path_monster)

        if verbose:
            print('Transformed shard '+str(htmid))
