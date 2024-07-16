import esutil
import os
import numpy as np
import lsst.utils

from .splinecolorterms import ColortermSpline
from .refcats import (GaiaXPInfo, GaiaXPuInfo, GaiaDR3Info, SkyMapperInfo, PS1Info,
                      VSTInfo, DESInfo, SynthLSSTInfo, LATISSInfo, SDSSuInfo,
                      FLAG_DICT)
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
    # Name of ID to use for matching.
    match_id_name = "GaiaDR3_id"

    """Assemble the Monster catalog.

    This class will copy Gaia (DR3) columns, and use the associated
    photometry from the DES-calibrated catalogs. The catalogs are read
    in reverse priority order so that the final fluxes for any object
    come from the top priority catalog. At the end the intermediate
    calibrated catalogs are converted to the final fluxes (e.g. LSST).

    Parameters
    ----------
    gaia_reference_class : `RefcatInfo`
        The input Gaia DR3 RefcatInfo object.
    catalog_info_class_list : `list` [`RefcatInfo`]
        Reverse-priority list of catalog info classes for assembly.
    monster_path_inp : `str`, optional
        Output monster path, overriding the class config.
    testing_mode : `bool`, optional
        Enter testing mode for read_stars?
    synth_system : `str`, optional
        Synthetic system to do final conversion.
    """
    def __init__(self,
                 gaia_reference_class=GaiaDR3Info,
                 catalog_info_class_list=[VSTInfo, SkyMapperInfo,
                                          PS1Info, GaiaXPInfo, DESInfo, GaiaXPuInfo],
                 monster_path_inp=None,
                 testing_mode=False,
                 target_catalog_info_class_list=[SynthLSSTInfo, LATISSInfo, DESInfo, SDSSuInfo],
                 do_u_band_slr=True,

                 ):

        self.gaia_reference_info = gaia_reference_class()
        self.catalog_info_class_list = [cat_info() for cat_info
                                        in catalog_info_class_list]
        self.target_catalog_info_class_list = [cat_info() for cat_info
                                               in target_catalog_info_class_list]

        self.testing_mode = testing_mode
        # will only create monster refcat for these bands
        self.all_bands = ['u', 'g', 'r', 'i', 'z', 'y']
        # Default path to write the outputs:
        self.monster_path_inp = monster_path_inp

        self.colorterm_path = os.path.join(lsst.utils.getPackageDir('the_monster'), 'colorterms')
        self.do_u_band_slr = do_u_band_slr

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

        target_systems = []
        target_systems_u_transform = []
        for cat_info_target in self.target_catalog_info_class_list:
            target_system_name = cat_info_target.NAME
            # for each target catalog, create columns for each band
            # that intersects with self.all_bands
            target_bands = set(self.all_bands).intersection(cat_info_target.bands)
            # create all tuples of target catalog band
            for band in target_bands:
                if (band == "u") & (target_system_name != "SDSS"):
                    # for u band where we are not using null transform
                    # (null is SDSS to SDSS) we want to create monster and then
                    # use monster DES g and r band to transform to u band
                    target_systems_u_transform.append((target_system_name, band))
                else:
                    target_systems.append((target_system_name, band))

        # get set of target system band pairs
        target_systems = sorted(list(set(target_systems)))
        target_systems_u_transform = sorted(list(set(target_systems_u_transform)))

        # create output columns
        for target_system_name, band in target_systems + target_systems_u_transform:

            gaia_stars_all.add_column(nan_column,
                                      name=f"monster_{target_system_name}_{band}_flux")
            gaia_stars_all.add_column(nan_column,
                                      name=f"monster_{target_system_name}_{band}_fluxErr")
            gaia_stars_all.add_column(int_column,
                                      name=f"monster_{target_system_name}_{band}_source_flag")

        if ("u" in self.all_bands) & ((len(target_systems_u_transform) > 0) | self.do_u_band_slr) \
                & ~("g" in self.all_bands) & ~("r" in self.all_bands):
            raise ValueError("u band transformations or slr require g and r bands to be in the bands list")

        # Loop over the refcats for griz bands
        for cat_info in self.catalog_info_class_list:
            # catalog_info_class_list should be a list of
            # cat_info = self.CatInfoClass() e.g. gaia cat

            # get set of bands for each catalog
            bands = set(self.all_bands).intersection(cat_info.bands)

            # Read in star cat that has already been transformed to the DES
            # system (if it exists).
            if os.path.isfile(cat_info.transformed_path+'/'+str(htmid)+'.fits'):
                cat_stars = read_stars(cat_info.transformed_path, [htmid],
                                       allow_missing=self.testing_mode)
            else:
                continue

            # for each band do transformations skip u band
            for target_system_name, band in target_systems:
                if band in bands:
                    if band == "u":
                        if target_system_name not in ['SDSS']:
                            raise ValueError(
                                "this portion of the code should only be the null transform for u band"
                            )
                        # since this is the null transform we dont need to
                        # apply a spline. I am struggling to figure out how
                        # to get the gaiaxpu in since it either has only
                        # u-band or will propogate other bands, open to input.
                        orig_flux = cat_stars[cat_info.get_transformed_flux_field(band)]
                        orig_flux_err = cat_stars[cat_info.get_transformed_flux_field(band)+'Err']
                        model_flux = orig_flux
                        model_flux_err = orig_flux_err

                    else:
                        # Transform from the DES to the synthetic system:
                        colorterm_file_string = 'DES_to_'+str(target_system_name)+'_band'
                        colorterm_spline = self.get_colorterm_spline(colorterm_file_string, band)
                        # apply colorterms to transform to target system mag
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
                    cat_stars[f"monster_{target_system_name}_{band}_flux"] = model_flux
                    cat_stars[f"monster_{target_system_name}_{band}_fluxErr"] = model_flux_err

                    # Apply selection to only apply transformations within the
                    # useful color range. (Note that the input DES-system
                    # catalogs already had their survey-specific cuts applied,
                    # so these cuts should be for the Synth{synth_system}
                    # transformations.)
                    # To DO: figure out what is going on here
                    # color_range = output_system.get_color_range(band)
                    # colors =
                    # cat_info.get_transformed_mag_colors(cat_stars, band)
                    # selected =
                    # (colors >= color_range[0]) & (colors <= color_range[1])
                    flux_not_nan = np.isfinite(cat_stars[f"monster_{target_system_name}_{band}_flux"])
                    if band == "u":
                        # if u band we are only doing null transform
                        # in this part. So, no color selection.
                        flag = flux_not_nan
                    else:
                        colors = cat_info.get_transformed_mag_colors(cat_stars, band)
                        selected = (colors >= -10) & (colors <= 10)
                        flag = selected & flux_not_nan

                    cat_stars_selected = cat_stars[flag]

                    # Skip cases that have no entries in cat_stars_selected
                    # (for whatever reason).
                    if len(cat_stars_selected) > 0:
                        # Match the transformed catalog to Gaia.
                        idx1, idx2 = esutil.numpy_util.match(gaia_stars_all['id'],
                                                             cat_stars_selected['GaiaDR3_id'])

                        # If the flux measurement is OK, write it to the
                        # overall Gaia catalog:
                        flux_col = f"monster_{target_system_name}_{band}_flux"
                        gaia_stars_all[flux_col][idx1] = cat_stars_selected[flux_col][idx2]
                        fluxerr_col = flux_col+'Err'
                        gaia_stars_all[fluxerr_col][idx1] = cat_stars_selected[fluxerr_col][idx2]

                        # Update the flags to denote which survey the flux came
                        # from:

                        gaia_stars_all[
                            f"monster_{target_system_name}_{band}_source_flag"
                        ][idx1] = cat_info.flag

        if self.do_u_band_slr & ("u" in self.all_bands):
            # for u band slr we use des g and r bands to
            # transform to sdss u band
            target_system_name = 'SDSS'
            band = "u"
            # we should have all of the sdss u band in monster
            # format at this point
            colorterm_file_string = 'DES_to_SDSS_band'
            colorterm_spline = self.get_colorterm_spline(colorterm_file_string, band)
            # apply colorterms to transform to target system mag

            flux_col = f'monster_DES_{colorterm_spline.source_field}_flux'
            flux_col_1 = f'monster_DES_{colorterm_spline.source_color_field_1}_flux'
            flux_col_2 = f'monster_DES_{colorterm_spline.source_color_field_2}_flux'

            orig_flux = gaia_stars_all[flux_col]
            orig_flux_err = gaia_stars_all[flux_col + 'Err']
            model_flux = colorterm_spline.apply(
                gaia_stars_all[flux_col_1],
                gaia_stars_all[flux_col_2],
                orig_flux,
            )
            # Rescale flux error to keep S/N constant
            model_flux_err = model_flux * (orig_flux_err/orig_flux)

            # not sure what colors should be here
            # colors = cat_info.get_transformed_mag_colors(cat_stars, band)
            # selected = (colors >= -10) & (colors <= 10)

            # only want to insert values that are not nan and are nan
            # in gaia_stars_all
            model_flux_not_nan = np.isfinite(model_flux)
            monster_flag_not_set = gaia_stars_all[f"monster_{target_system_name}_{band}_source_flag"] == -1
            flag = model_flux_not_nan & monster_flag_not_set
            gaia_stars_all[flux_col][flag] = model_flux[flag]
            gaia_stars_all[flux_col + 'Err'][flag] = model_flux_err[flag]
            # Update the flags to denote which survey the flux came
            # from:
            gaia_stars_all[f"monster_{target_system_name}_{band}_source_flag"][flag] = FLAG_DICT["SLR"]

        if len(target_systems_u_transform) > 0:
            for target_system_name, band in target_systems_u_transform:
                colorterm_file_string = 'SDSS_to_'+str(target_system_name)+'_band'
                colorterm_spline = self.get_colorterm_spline(colorterm_file_string, band)
                # apply colorterms to transform to target system mag
                flux_col = f'monster_SDSS_{colorterm_spline.source_field}_flux'
                flux_col_1 = f'monster_DES_{colorterm_spline.source_color_field_1}_flux'
                flux_col_2 = f'monster_DES_{colorterm_spline.source_color_field_2}_flux'

                orig_flux = gaia_stars_all[flux_col]
                orig_flux_err = gaia_stars_all[flux_col + 'Err']
                orig_flags = gaia_stars_all["monster_SDSS_u_source_flag"]

                model_flux = colorterm_spline.apply(
                    gaia_stars_all[flux_col_1],
                    gaia_stars_all[flux_col_2],
                    orig_flux,
                )

                # Rescale flux error to keep S/N constant
                model_flux_err = model_flux * (orig_flux_err/orig_flux)

                model_flux_not_nan = np.isfinite(model_flux)
                flag = model_flux_not_nan  # & selected

                # Add the fluxes and their errors to the catalog:
                gaia_stars_all[f"monster_{target_system_name}_{band}_flux"][flag] = model_flux[flag]
                gaia_stars_all[f"monster_{target_system_name}_{band}_fluxErr"][flag] = model_flux_err[flag]

                # not sure what this should be here
                # colors =
                # cat_info.get_transformed_mag_colors(gaia_stars_all, band)
                # selected =
                # (colors >= -10) & (colors <= 10)

                # I think new flags should be the same as sdss u band flags
                gaia_stars_all[f"monster_{target_system_name}_{band}_source_flag"] = orig_flags

        if self.monster_path_inp is None:
            monster_path = "/sdf/data/rubin/shared/the_monster/sharded_refcats/monster_v2"
        else:
            monster_path = self.monster_path_inp

        # Output the finished catalog for the shard:
        os.makedirs(monster_path, exist_ok=True)
        output_file = os.path.join(monster_path, f"{htmid}.fits")

        # Convert the refcat to a SimpleCatalog
        monsterSchema = makeMonsterSchema(gaia_stars_all.itercols(),
                                          target_systems=target_systems + target_systems_u_transform)
        monsterCat = makeMonsterCat(monsterSchema, gaia_stars_all)

        # Save the shard to FITS.
        monsterCat.writeFits(output_file)

        if verbose:
            print('Transformed shard '+str(htmid))

    def get_colorterm_spline(self, colorterm_file_string, band):
        colorterm_filename = os.path.join(
            self.colorterm_path,
            colorterm_file_string+f'_{band}.yaml',
        )
        assert os.path.isfile(colorterm_filename), f"File {colorterm_filename} not found."

        # read in spline
        colorterm_spline = ColortermSpline.load(colorterm_filename)

        return colorterm_spline
