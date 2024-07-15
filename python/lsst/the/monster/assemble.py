import esutil
import os
import numpy as np
import lsst.utils

from .splinecolorterms import ColortermSpline
from .refcats import (GaiaXPInfo, GaiaXPuInfo, GaiaDR3Info, SkyMapperInfo, PS1Info,
                      VSTInfo, DESInfo, SynthLSSTInfo, LATISSInfo, SDSSuInfo)
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
                 ):

        self.gaia_reference_info = gaia_reference_class()
        self.catalog_info_class_list = [cat_info() for cat_info
                                        in catalog_info_class_list]
        self.target_catalog_info_class_list = [cat_info() for cat_info
                                        in target_catalog_info_class_list]
    
        self.testing_mode = testing_mode
        # will only create monster refcat for these bands
        self.all_bands = ['u','g', 'r', 'i', 'z', 'y']
        # Default path to write the outputs:
        self.monster_path_inp = monster_path_inp

        self.colorterm_path = os.path.join(lsst.utils.getPackageDir('the_monster'), 'colorterms')

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

        
        target_systems =[]
        for cat_info_target in self.target_catalog_info_class_list:
            target_system_name = cat_info_target.NAME
            # for each target catalog, create columns for each band that intersects with self.all_bands
            target_bands = set(self.all_bands).intersection(cat_info_target.bands)
            # create all tuples of target catalog band
            for band in target_bands:
                target_systems.append((target_system_name, band))
        
        # get set of target system band pairs
        target_systems = sorted(list(set(target_systems)))
        
        # create output columns
        for target_system_name, band in target_systems:
            
            gaia_stars_all.add_column(nan_column,
                                    name=f"monster_{target_system_name}_{band}_flux")
            gaia_stars_all.add_column(nan_column,
                                    name=f"monster_{target_system_name}_{band}_fluxErr")
            gaia_stars_all.add_column(int_column,
                                    name=f"monster_{target_system_name}_{band}_source_flag")
        
        print(gaia_stars_all.columns)
        # Loop over the refcats
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
            
            # for each band do transformations
            for target_system_name,band in target_systems:
                if band in bands:
                    if band == "u":
                        # for u band, target system should be sdss and we will either be
                        # currently u band didn't make it into gaiaxpu transformed
                        # TODO: change isolate and transform to include u band
                        colorterm_file_string = str(cat_info.name) + '_to_'+str(target_system_name)+'_band'
                        
                    else:
                        # Transform from the DES to the synthetic system:
                        colorterm_file_string = 'DES_to_'+str(target_system_name)+'_band'
                    
                    colorterm_filename = os.path.join(
                        self.colorterm_path,
                        colorterm_file_string+f'_{band}.yaml',
                    )
                    assert os.path.isfile(colorterm_filename), f"File {colorterm_filename} not found."

                    # read in spline
                    colorterm_spline = ColortermSpline.load(colorterm_filename)

                    # apply colorterms to transform to Synth{synth_system} mag
                    band_1, band_2 = cat_info.get_color_bands(band)
                    try:
                        orig_flux = cat_stars[cat_info.get_transformed_flux_field(band)]
                    except:
                        import pdb; pdb.set_trace()
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
                    # colors = cat_info.get_transformed_mag_colors(cat_stars, band)
                    # selected = (colors >= color_range[0]) & (colors <= color_range[1])

                    colors = cat_info.get_transformed_mag_colors(cat_stars, band)
                    selected = (colors >= -10) & (colors <= 10)

                    flux_not_nan = np.isfinite(cat_stars[f"monster_{target_system_name}_{band}_flux"])
                    flag = selected & flux_not_nan
                    cat_stars_selected = cat_stars[flag]

                    # Skip cases that have no entries in cat_stars_selected
                    # (for whatever reason).
                    if len(cat_stars_selected) > 0:
                        # Match the transformed catalog to Gaia.
                        a, b = esutil.numpy_util.match(gaia_stars_all['id'],
                                                        cat_stars_selected['GaiaDR3_id'])

                        # If the flux measurement is OK, write it to the
                        # overall Gaia catalog:
                        flux_col = f"monster_{target_system_name}_{band}_flux"
                        gaia_stars_all[flux_col][a] = cat_stars_selected[flux_col][b]
                        fluxerr_col = flux_col+'Err'
                        gaia_stars_all[fluxerr_col][a] = cat_stars_selected[fluxerr_col][b]

                        # Update the flags to denote which survey the flux came
                        # from:
                        gaia_stars_all[f"monster_{target_system_name}_{band}_source_flag"][a] = cat_info.flag
        
        # run slr u band bit 
        import pdb; pdb.set_trace()
        if 'u' in self.all_bands:
            band = 'u'
            cat_info = SDSSuInfo()
        #   self.add_slr_to_sdss_u()
        # self.transfrom_sdss_u_to_monster()

        if self.monster_path_inp is None:
            monster_path = "/sdf/data/rubin/shared/the_monster/sharded_refcats/monster_v1"
        else:
            monster_path = self.monster_path_inp

        # Output the finished catalog for the shard:
        os.makedirs(monster_path, exist_ok=True)
        output_file = os.path.join(monster_path, f"{htmid}.fits")

        # Convert the refcat to a SimpleCatalog
        monsterSchema = makeMonsterSchema(gaia_stars_all.itercols(), self.all_bands,
                                          output_system=output_system)
        monsterCat = makeMonsterCat(monsterSchema, gaia_stars_all)

        # Save the shard to FITS.
        monsterCat.writeFits(output_file)

        if verbose:
            print('Transformed shard '+str(htmid))
    # def transform_to_sdss_u(output_system):
         
    #     if band in output_system.bands:
    #         if band == 'u' & cat_info.NAME == 'GaiaXPu':
    #             colorterm_file_string = 'foo'
    #         elif band == 'u' & cat_info.NAME == 'SDSSu':
    #             colorterm_file_string = 'foo'
    #             continue
    #         else:
                            