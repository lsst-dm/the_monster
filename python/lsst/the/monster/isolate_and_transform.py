import os
from smatch import Matcher
import numpy as np

from lsst.pipe.tasks.isolatedStarAssociation import IsolatedStarAssociationTask
from .splinecolorterms import ColortermSpline
from .refcats import GaiaXPInfo, GaiaDR3Info, SkyMapperInfo, PS1Info, VSTInfo, DESInfo, GaiaXPuInfo
from .utils import read_stars, makeRefSchema, makeRefCat

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
    gaia_reference_class : `RefCatInfo`
        The input Gaia DR3 RefcatInfo object.
    catalog_info_class_list : `list` [`RefcatInfo`]
        List of RefcatInfo objects for catalogs to transform.
    transformed_path_inp : `str`, optional
        The path to write the outputs to.
    testing_mode : `bool`
        Enter testing mode for read_stars?
    """

    def __init__(self,
                 gaia_reference_class=GaiaDR3Info,
                 catalog_info_class_list=[GaiaXPInfo, SkyMapperInfo,
                                          PS1Info, VSTInfo, DESInfo,
                                          GaiaXPuInfo],
                 transformed_path_inp=None,
                 testing_mode=False,
                 ):

        self.gaia_reference_info = gaia_reference_class()
        self.catalog_info_class_list = [cat_info() for cat_info
                                        in catalog_info_class_list]
        self.testing_mode = testing_mode
        self.transformed_path_inp = transformed_path_inp

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

        # isolate the gaia cat
        gaia_stars = self._remove_neighbors(
            gaia_stars_all
        )
        # loop over other catalogs
        for cat_info in self.catalog_info_class_list:
            # catalog_info_class_list should be a list of
            # cat_info = self.CatInfoClass() e.g. gaia cat

            # output columns are target catalog id, gaia id, coordinates,
            # and the des (or sdss for u-band) fluxes
            outcols = ["id", self.gaia_reference_info.name + "_id", "coord_ra", "coord_dec"]
            outcols += [cat_info.get_transformed_flux_field(band) for band in cat_info.bands]
            outcols += [cat_info.get_transformed_flux_field(band) + "Err" for band in cat_info.bands]

            # read in star cat (if it exists)
            if os.path.isfile(cat_info.path+'/'+str(htmid)+'.fits'):
                cat_stars = read_stars(cat_info.path, [htmid], allow_missing=self.testing_mode)

                # match with gaia_stars
                with Matcher(cat_stars["coord_ra"], cat_stars["coord_dec"]) as m:
                    idx, i1, i2, d = m.query_knn(
                        gaia_stars["coord_ra"],
                        gaia_stars["coord_dec"],
                        distance_upper_bound=0.5/3600.0,
                        return_indices=True,
                    )
                cat_stars = cat_stars[i1]
                cat_stars.add_column(gaia_stars["id"][i2],
                                     name=self.gaia_reference_info.name + "_id")

                for band in cat_info.bands:
                    # yaml spline fits are per-band, so loop over bands
                    # read in spline
                    filename = cat_info.colorterm_file(band)
                    colorterm_spline = ColortermSpline.load(filename)

                    # apply colorterms to transform to des mag
                    band_1, band_2 = cat_info.get_color_bands(band)
                    orig_flux = cat_stars[cat_info.get_flux_field(band)]
                    orig_flux_err = cat_stars[cat_info.get_flux_field(band) + 'Err']
                    model_flux = colorterm_spline.apply(
                        cat_stars[cat_info.get_flux_field(band_1)],
                        cat_stars[cat_info.get_flux_field(band_2)],
                        orig_flux,
                    )

                    # Rescale flux error to keep S/N constant
                    model_flux_err = model_flux * (orig_flux_err/orig_flux)

                    # Append the modeled flux columns to cat_stars
                    cat_stars.add_column(model_flux,
                                         name=cat_info.get_transformed_flux_field(band))
                    cat_stars.add_column(model_flux_err,
                                         name=cat_info.get_transformed_flux_field(band) + 'Err')

                    # Apply selection to ensure that only useful stars have
                    # transformations.
                    selected = cat_info.select_stars(cat_stars, band)
                    cat_stars[cat_info.get_transformed_flux_field(band)][~selected] = np.nan
                    cat_stars[cat_info.get_transformed_flux_field(band) + 'Err'][~selected] = np.nan

                # If any stars are nans in all transformed filters, they should
                # be removed.
                n_measurements = np.zeros(len(cat_stars))
                for band in cat_info.bands:
                    n_measurements[np.isfinite(cat_stars[cat_info.get_transformed_flux_field(band)])] += 1
                cat_stars = cat_stars[n_measurements > 0]

                if self.transformed_path_inp is None:
                    transformed_path = cat_info.transformed_path
                else:
                    transformed_path = self.transformed_path_inp

                if os.path.exists(transformed_path) is False:
                    os.makedirs(transformed_path)
                output_file = os.path.join(transformed_path, f"{htmid}.fits")

                # Convert the refcat to a SimpleCatalog
                refSchema = makeRefSchema(cat_info,
                                          self.gaia_reference_info.name)
                refCat = makeRefCat(refSchema, cat_stars[outcols],
                                    cat_info,
                                    self.gaia_reference_info.name)

                # Save the shard to FITS.
                refCat.writeFits(output_file)

                if verbose:
                    print('Transformed '+cat_info.path+'/'+str(htmid)+'.fits')

            else:
                if verbose:
                    print(cat_info.path+'/'+str(htmid)+'.fits does not exist.')

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
        # Set isolation radius to 1.0 arcsec.
        isaTask.config.isolation_radius = 1.0

        # Remove the neighbors.
        return isaTask._remove_neighbors(catalog)
