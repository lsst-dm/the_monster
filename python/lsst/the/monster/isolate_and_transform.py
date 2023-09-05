import os
from smatch import Matcher

from lsst.pipe.tasks.isolatedStarAssociation import IsolatedStarAssociationTask

from lsst.the.monster.splinecolorterms import ColortermSpline
from lsst.the.monster.refcats import GaiaXPInfo, GaiaDR3Info, SkyMapperInfo, PS1Info, VSTInfo
from lsst.the.monster.utils import read_stars

__all__ = ["MatchAndTransform"]

"""
This Python script takes a sharded catalog and outputs a shard transformed to
DES bandpasses.

The following steps are performed:
1. Read in the Gaia shard.
2. Run the IsolatedStarAssociationTask on the Gaia shard.
3. Read in each catalog.
4. Match each catalog with the Gaia DR3 catalog.
5. For each band, interpolate the DES magnitudes.
6. Save the catalog shard.
"""


class MatchAndTransform:
    """Match catalogs to Gaia and transform them to 'the_monster'
       reference frame.

    Parameters
    ----------
    htmid : `int`
        htm level 7 id of catalog(s).
    """
    GaiaDR3CatInfoClass = GaiaDR3Info
    testing_mode = False

    def run(self,
            htmid=None,
            catalog_list=[GaiaXPInfo, SkyMapperInfo, PS1Info, VSTInfo],
            write_path_inp=None
        ):

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
            # catalog_list should be a list of
            # cat_info = self.CatInfoClass() e.g. gaia cat
            
            #output columns are target catalog id, gaia id, coordinates
            # and the des fluxes
            outcols = ["id", gaia_info.name + "_id", "coord_ra", "coord_dec"]
            outcols +=[f"decam_{band}_flux_from_{cat_info().name}" for band in cat_info().bands]
            import pdb; pdb.set_trace()
            # read in star cat (if it exists)
            if os.path.isfile(cat_info().path+'/'+str(htmid)+'.fits'):
                cat_stars = read_stars(cat_info().path, [htmid], allow_missing=self.testing_mode)

                # match with gaia_stars
                with Matcher(gaia_stars["coord_ra"], gaia_stars["coord_dec"]) as m:
                    idx, i1, i2, d = m.query_knn(
                        cat_stars["coord_ra"],
                        cat_stars["coord_dec"],
                        distance_upper_bound=0.5/3600.0,
                        return_indices=True,
                    )
                cat_stars = cat_stars[i2]
                cat_stars.add_column(gaia_stars["id"][i1], name=gaia_info.name + "_id")

                for band in cat_info.bands:
                    # yaml spline fits are per-band, so loop over bands
                    # read in spline
                    filename = os.path.abspath(os.path.dirname(__file__))[:-23]
                    filename += 'colorterms/'+cat_info().name
                    filename += '_to_DES_band_'+str(band)+'.yaml'

                    colorterm_spline = ColortermSpline.load(filename)

                    # apply colorterms to transform to des mag
                    band_1, band_2 = cat_info().get_color_bands(band)
                    model = colorterm_spline.apply(
                        cat_stars[cat_info().get_flux_field(band_1)],
                        cat_stars[cat_info().get_flux_field(band_2)],
                        cat_stars[cat_info().get_flux_field(band)],
                    )

                    model_flux = cat_stars[cat_info().get_flux_field(band)]/model
                    # Append the modeled mags column to cat_stars
                    cat_stars.add_column(model_flux, name=f"decam_{band}_flux_from_{cat_info().name}")

                if write_path_inp is None:
                    write_path = cat_info().path + '_transformed/'
                    if cat_info().name == 'PS1':
                        write_path = '/sdf/data/rubin/shared/the_monster/sharded_refcats/ps1_transformed'
                else:
                    write_path = write_path_inp

                if os.path.exists(write_path) is False:
                    os.makedirs(write_path)
                write_path += f"/{htmid}.fits"

                # Save the shard to FITS.
                
                cat_stars[outcols].write(write_path, overwrite=True)

            else:
                print(cat_info().path+'/'+str(htmid)+'.fits does not exist.')

    def _remove_neighbors(self, catalog):
        """Removes neighbors from a catalog.

        Args:
            catalog (pandas.DataFrame): The catalog to remove neighbors from.

        Returns:
            pandas.DataFrame: The catalog with neighbors removed.
        """

        # Create an instance of the IsolatedStarAssociationTask class.
        isaTask = IsolatedStarAssociationTask()

        # Set the RA and Dec columns.
        isaTask.config.ra_column = "coord_ra"
        isaTask.config.dec_column = "coord_dec"

        # Remove the neighbors.
        return isaTask._remove_neighbors(catalog)
