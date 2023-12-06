import esutil
import os
import numpy as np
import lsst.utils

from .splinecolorterms import ColortermSpline
from .refcats import (GaiaXPInfo, GaiaDR3Info, SkyMapperInfo, PS1Info,
                      VSTInfo, DESInfo, SynthLSSTInfo)
from .utils import read_stars, makeMonsterSchema, makeMonsterCat

__all__ = ["AssembleMonsterRefcat"]

"""
This Python script starts with the full Gaia DR3 catalog, then assembles a
monster refcat by doing the following:

For each shard:
1. Read the full Gaia DR3 catalog for the shard
2. Initialize 18 columns (ugrizy fluxes and their errors, and a source flag
    (integer)) for the results
(3-6): Within a loop over surveys in order from lowest to highest priority:
3. Read each of the (already transformed to the DES system) refcats for the
    shard
4. Transform each refcat to the (synthetic) system (e.g., LSST or LATISS)
5. Match each refcat to the Gaia DR3 catalog
6. Update the fluxes, flux errors, and flags whenever a value is non-NaN
"""


class AssembleMonsterRefcat:
    def __init__(self,
                 gaia_reference_class=GaiaDR3Info,
                 catalog_info_class_list=[VSTInfo, SkyMapperInfo,
                                          PS1Info, GaiaXPInfo, DESInfo],
                 write_path_inp=None,
                 testing_mode=False,
                 synth_system='LSST',
                 ):

        self.gaia_reference_info = gaia_reference_class()
        self.catalog_info_class_list = [cat_info() for cat_info
                                        in catalog_info_class_list]
        # Dict with lookup info for different synthetic systems. To add a new
        # system, add its "Info" class to refcats.py and import it here,
        # and add an entry to this dict to point to that info class.
        synth_info_dict = {'LSST': SynthLSSTInfo}
        self.synth_system = synth_system
        self.synth_info = synth_info_dict[self.synth_system]()
        self.testing_mode = testing_mode
        self.write_path_inp = write_path_inp
        self.all_bands = ['u', 'g', 'r', 'i', 'z', 'y']
        # Default path to write the outputs:
        self.write_path_monster = "/sdf/data/rubin/shared/the_monster/sharded_refcats/monster_v1"

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

        output_system = str.lower(self.synth_system)

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
                                      name=f"monster_{output_system}_{band}_flux")
            gaia_stars_all.add_column(nan_column,
                                      name=f"monster_{output_system}_{band}_fluxErr")
            gaia_stars_all.add_column(int_column,
                                      name=f"monster_{output_system}_{band}_source_flag")

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

                    # Transform from the DES to the synthetic system:
                    colorterm_path = os.path.join(
                        lsst.utils.getPackageDir("the_monster"),
                        "colorterms",
                    )
                    colorterm_file_string = 'DES_to_Synth'+str.upper(self.synth_system)+'_band'
                    colorterm_filename = os.path.join(
                        colorterm_path,
                        colorterm_file_string+f'_{band}.yaml',
                    )

                    # read in spline
                    colorterm_spline = ColortermSpline.load(colorterm_filename)

                    # apply colorterms to transform to Synth{synth_system} mag
                    band_1, band_2 = cat_info.get_color_bands(band)
                    orig_flux = cat_stars[cat_info.get_transformed_flux_field(band)]
                    orig_flux_err = cat_stars[cat_info.get_transformed_flux_field(band)+'Err']
                    model_flux = colorterm_spline.apply(
                        cat_stars[cat_info.get_transformed_flux_field(band_1)],
                        cat_stars[cat_info.get_transformed_flux_field(band_2)],
                        orig_flux,
                    )

                    # Rescale flux error to keep S/N constant
                    model_flux_err = model_flux * (orig_flux_err/orig_flux)

                    # Add the fluxes and their errors to the catalog:
                    cat_stars[f"monster_{output_system}_{band}_flux"] = model_flux
                    cat_stars[f"monster_{output_system}_{band}_fluxErr"] = model_flux_err

                    # Apply selection to only apply transformations within the
                    # useful color range. (Note that the input DES-system
                    # catalogs already had their survey-specific cuts applied,
                    # so these cuts should be for the Synth{synth_system}
                    # transformations.)
                    color_range = self.synth_info.get_color_range(band)
                    colors = cat_info.get_transformed_mag_colors(cat_stars, band)
                    selected = (colors >= color_range[0]) & (colors <= color_range[1])

                    flux_not_nan = np.isfinite(cat_stars[f"monster_{output_system}_{band}_flux"])
                    flag = selected & flux_not_nan
                    cat_stars_selected = cat_stars[flag]

                    # Match the transformed catalog to Gaia.
                    a, b = esutil.numpy_util.match(gaia_stars_all['id'],
                                                   cat_stars_selected['GaiaDR3_id'])

                    # If the flux measurement is OK, write it to the overall
                    # Gaia catalog:
                    flux_col = f"monster_{output_system}_{band}_flux"
                    gaia_stars_all[flux_col][a] = cat_stars_selected[flux_col][b]
                    fluxerr_col = flux_col+'Err'
                    gaia_stars_all[fluxerr_col][a] = cat_stars_selected[fluxerr_col][b]

                    # Update the flags to denote which survey the flux came
                    # from:
                    gaia_stars_all[f"monster_{output_system}_{band}_source_flag"][a] = cat_info.flag

        if self.write_path_inp is None:
            write_path = self.write_path_monster
        else:
            write_path = self.write_path_inp

        # Output the finished catalog for the shard:
        os.makedirs(write_path, exist_ok=True)
        write_path += f"/{htmid}.fits"

        # Convert the refcat to a SimpleCatalog
        monsterSchema = makeMonsterSchema(gaia_stars_all.itercols(), self.all_bands,
                                          output_system=output_system)
        monsterCat = makeMonsterCat(monsterSchema, gaia_stars_all)

        # Save the shard to FITS.
        monsterCat.writeFits(write_path)

        if verbose:
            print('Transformed shard '+str(htmid))
