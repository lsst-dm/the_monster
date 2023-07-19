import numpy as np
import fitsio
import os
from smatch.matcher import Matcher
import warnings
from astropy.utils.exceptions import AstropyWarning

from lsst.meas.algorithms.readFitsCatalogTask import ReadFitsCatalogTask
from lsst.pipe.tasks.isolatedStarAssociation import IsolatedStarAssociationTask
import lsst.pex.config as pexConfig

# I think I want to create seperate folders for each catalog and have a yaml
# file in the super directory that specifies what columns to use for each 
# catalog.
class RefCatMatchConfig(pexConfig.Config):
    """Configuration for IsolatedStarAssociationTask."""

    path = pexConfig.Field(
        doc=("Path to catalog directory"),
        dtype=str,
        default=None,
    )
    catalog_name = pexConfig.Field(
        doc="catalog name to be used for filepath e.g. gaiaDr3",
        dtype=str,
        default="gaiaDr3",
    )
    id_column = pexConfig.Field(
        doc="Name of column with source id.",
        dtype=str,
        default="id",
    )
    ra_column = pexConfig.Field(
        doc="Name of column with right ascension.",
        dtype=str,
        default="ra",
    )
    dec_column = pexConfig.Field(
        doc="Name of column with declination.",
        dtype=str,
        default="decl",
    )
    extra_columns = pexConfig.ListField(
        doc="Extra names of columns to read and persist Not currently implemented",
        dtype=str,
        default=[],
    )
    column_map = pexConfig.DictField[str, str](
        doc=(
            "Full name of instFlux field to use for s/n selection and persistence. "
            "The associated flag will be implicity included in bad_flags. "
            "Note that this is expected to end in ``instFlux``."
        ),
        keytype=str,
        itemtype=str,
        default={"coord_ra": "ra", "coord_dec": "decl"},
    )
    match_radius = pexConfig.Field(
        doc="Match radius (arcseconds)",
        dtype=float,
        default=1.0,
    )


class CreateAndMatchIsolatedGaiaCatConfig(pexConfig.Config):
    """Configuration for IsolatedStarAssociationTask."""

    isolation_radius = pexConfig.Field(
        doc=(
            "Isolation radius (arcseconds).  Any stars with average centroids "
            "within this radius of another star will be rejected from the final "
            "catalog.  This radius should be at least 2x match_radius."
        ),
        dtype=float,
        default=2.0,
    )
    # could check if we have already run the isolated cat creator
    primary_catalog_name = pexConfig.Field(
        doc="primary catalog name to be used for filepath e.g. gaiaDr3",
        dtype=str,
        default="gaiaDr3",
    )
    primary_catalog_config = pexConfig.ConfigField(
        doc="config for primary catalog to create isolated cata and match with other catalogs",
        dtype=RefCatMatchConfig,
        default=None,
    )
    matched_catalog_configs = pexConfig.ConfigDictField(
        doc="A dict of configs describing the catalogs to match with the isolated gaia catalog.",
        keytype=str,
        itemtype=RefCatMatchConfig,
        default={},
    )
    primary_force = pexConfig.Field(
        doc="force recreating the primary catalog",
        dtype=bool,
        default=False,
    )
    out_path = pexConfig.Field(
        doc=("output path for matched catalogs"),
        dtype=str,
        default="/sdf/group/rubin/g/project/the_monster/matched_catalogs/",
    )


class CreateAndMatchIsolatedGaiaCat:
    ConfigClass = CreateAndMatchIsolatedGaiaCatConfig

    def __init__(self, configfile):
        # self.config=CreateAndMatchIsolatedGaiaCatConfig()
        self.config = CreateAndMatchIsolatedGaiaCat.ConfigClass()
        self.config.load(configfile)

        os.makedirs(
            self.config.out_path
            + self.config.primary_catalog_config.catalog_name
            + "_primary",
            exist_ok=True,
        )
        for key in self.config.matched_catalog_configs:
            os.makedirs(
                self.config.out_path
                + self.config.matched_catalog_configs[key].catalog_name,
                exist_ok=True,
            )

    def _remove_neighbors(self, catalog, cat_config):
        isaTask = IsolatedStarAssociationTask()
        isaTask.config.ra_column = cat_config.ra_column
        isaTask.config.dec_column = cat_config.dec_column
        return isaTask._remove_neighbors(catalog)
    def _find_catalogs(self):
        if os.path
    def _read_catalog(self, htmid, cat_config):
        catalog_loader = ReadFitsCatalogTask()
        catalog_loader.config.column_map = cat_config.column_map
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", AstropyWarning)
            catalog = catalog_loader.run(cat_config.path + htmid + ".fits")
        # convert ra/dec from radians to degrees
        catalog[cat_config.ra_column] = np.degrees(catalog[cat_config.ra_column])
        catalog[cat_config.dec_column] = np.degrees(catalog[cat_config.dec_column])
        return catalog

    def run(self, htmid):
        # htmid
        write_path = (
            self.config.out_path
            + self.config.primary_catalog_config.catalog_name
            + "_primary/"
        )
        write_path += f"{self.config.primary_catalog_config.catalog_name}_{htmid}.fits"
        if ~os.path.exists(write_path) | (self.config.primary_force is True):
            primary_cat = self._read_catalog(htmid, self.config.primary_catalog_config)
            primary_isolated_cat = self._remove_neighbors(
                primary_cat, self.config.primary_catalog_config
            )
            fitsio.write(write_path, primary_isolated_cat, clobber=True)
        else:
            # change to use read catalog method
            primary_isolated_cat = fitsio.read(write_path)
        miss_num = 0
        miss_list = []
        with Matcher(
            primary_isolated_cat[self.config.primary_catalog_config.ra_column],
            primary_isolated_cat[self.config.primary_catalog_config.dec_column],
        ) as matcher:
            for key in self.config.matched_catalog_configs:
                cat_config = self.config.matched_catalog_configs[key]
                try:
                    cat = self._read_catalog(htmid, cat_config)
                except:
                    # print(f"no file {key}: {htmid}")
                    miss_num += 1
                    miss_list.append(key)
                    continue
                idx, i1, i2, distance = matcher.query_radius(
                    cat[cat_config.ra_column],
                    cat[cat_config.dec_column],
                    cat_config.match_radius / 3600,
                    return_indices=True,
                )

                sel = distance.filled() < cat_config.match_radius / 3600
                matched_cat_dtype = cat.dtype.descr + [("primary_id", "int64")]
                matched_cat = np.recarray(sel.sum(), dtype=matched_cat_dtype)
                matched_cat[list(cat.dtype.names)] = cat[i2[sel]]
                matched_cat["primary_id"] = primary_isolated_cat[
                    self.config.primary_catalog_config.id_column
                ][i1[sel]].filled()
                write_path = self.config.out_path + cat_config.catalog_name + "/"
                write_path += f"{cat_config.catalog_name}_{htmid}.fits"
                fitsio.write(write_path, matched_cat, clobber=True)
        print(
            f"{htmid} missing {miss_num}/{len(self.config.matched_catalog_configs.keys())}: {miss_list}"
        )
