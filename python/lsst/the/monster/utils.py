import os
from astropy import units
import numpy as np

from lsst.afw.table import SimpleCatalog
import lsst.afw.table as afwTable

__all__ = ["read_stars", "makeRefSchema", "makeRefCat",
           "makeMonsterSchema", "makeMonsterCat"]


def read_stars(path, indices, allow_missing=False):
    """Read stars from a sharded catalog.

    Parameters
    ----------
    path : `str`
        Path to sharded catalog.
    indices : `list` [`int`]
        List of pixel indices.
    allow_missing : `bool`, optional
        Allow missing pixels?  Used for testing.

    Returns
    -------
    catalog : `astropy.Table`
        Astropy table catalog.
    """
    stars = None
    for index in indices:
        fname = os.path.join(path, str(index) + ".fits")
        if not os.path.isfile(fname):
            if allow_missing:
                continue
        try:
            temp = SimpleCatalog.readFits(fname)
        except RuntimeError as e:
            if allow_missing:
                continue
            else:
                raise e
        if stars is None:
            stars = temp
        else:
            stars.extend(temp)

    if stars is None:
        return []

    stars = stars.copy(deep=True).asAstropy()

    stars["coord_ra"].convert_unit_to(units.degree)
    stars["coord_dec"].convert_unit_to(units.degree)

    return stars


def makeRefSchema(survey, bands, reference_name):
    """
    Make the refcat schema

    Parameters
    ----------
    survey : `str`
        Name of the survey whose refcat we are writing
    bands : `List` of `str`
        Names of the bands in the refcat
    reference_name : `str`
        Name of the overall reference catalog

    Returns
    -------
    refSchema: `lsst.afw.table.Schema`
    """

    refSchema = afwTable.SimpleTable.makeMinimalSchema()
    refSchema.addField(reference_name+'_id', type='L', doc='Gaia DR3 ID')

    for band in bands:
        colname = 'decam_'+band+'_from_'+survey+'_flux'
        colname_err = colname+'Err'
        refSchema.addField(colname, type='D',
                           doc='flux transformed to DECam system',
                           units='nJy')
        refSchema.addField(colname_err, type='D',
                           doc='error on flux transformed to DECam system',
                           units='nJy')

    return refSchema


def makeRefCat(refSchema, refTable, survey, bands, reference_name):
    """
    Make the standard star catalog for persistence

    Parameters
    ----------
    refSchema: `lsst.afw.table.Schema`
       Standard star catalog schema
    refTable: `Astropy Table`
       Reference catalog to convert
    survey : `str`
        Name of the survey whose refcat we are writing
    bands : `List` of `str`
        Names of the bands in the refcat
    reference_name : `str`
        Name of the overall reference catalog

    Returns
    -------
    refCat: `lsst.afw.table.BaseCatalog`
       Standard star catalog for persistence
    """

    refCat = afwTable.SimpleCatalog(refSchema)
    refCat.resize(np.size(refTable))

    refCat['id'][:] = refTable['id']
    refCat[reference_name+'_id'][:] = refTable[reference_name+'_id']
    refCat['coord_ra'][:] = np.deg2rad(refTable['coord_ra'])
    refCat['coord_dec'][:] = np.deg2rad(refTable['coord_dec'])

    for band in bands:
        colname = 'decam_'+band+'_from_'+survey+'_flux'
        colname_err = colname+'Err'
        refCat[colname][:] = refTable[colname]
        refCat[colname_err][:] = refTable[colname_err]

    return refCat

def makeMonsterSchema(gaia_catalog_columns, bands):
    """
    Make the monster refcat schema. Include all columns from Gaia, as well
    as transformed fluxes, flux errors, and flags identifying the source
    of each flux entry.

    Parameters
    ----------
    gaia_catalog_columns : `List` of `TableColumns`
        Gaia catalog columns (e.g., from "gaia_stars_all.itercols()")
    bands : `List` of `str`
        Names of the bands to include in the monster refcat

    Returns
    -------
    monsterSchema: `lsst.afw.table.Schema`
    """

    monsterSchema = afwTable.SimpleTable.makeMinimalSchema()

    exclude_columns = ["id", "coord_ra", "coord_dec", "monster_lsst_u_flux",
                       "monster_lsst_u_fluxErr", "monster_lsst_u_source_flag",
                       "monster_lsst_g_flux", "monster_lsst_g_fluxErr",
                       "monster_lsst_g_source_flag", "monster_lsst_r_flux",
                       "monster_lsst_r_fluxErr", "monster_lsst_r_source_flag",
                       "monster_lsst_i_flux", "monster_lsst_i_fluxErr",
                       "monster_lsst_i_source_flag", "monster_lsst_z_flux",
                       "monster_lsst_z_fluxErr", "monster_lsst_z_source_flag",
                       "monster_lsst_y_flux", "monster_lsst_y_fluxErr",
                       "monster_lsst_y_source_flag"]

    fieldtype_dict = {'float32':'F', 'float64':'D',
                      'int64':'L', 'bool':'B'}

    for col in gaia_catalog_columns:
        if col.name not in exclude_columns:
            if col.unit is not None:
                monsterSchema.addField(col.name,
                                       type=fieldtype_dict[col.dtype.name],
                                       doc=col.description,
                                       units=col.unit.to_string()
                                       )
            else:
                monsterSchema.addField(col.name,
                                       type=fieldtype_dict[col.dtype.name],
                                       doc=col.description,
                                       units=''
                                       )


    for band in bands:
        fluxcolname = f"monster_lsst_{band}_flux"
        fluxcolname_err = fluxcolname+'Err'
        flagcolname = f"monster_lsst_{band}_source_flag"
        monsterSchema.addField(fluxcolname, type='D',
                               doc='flux transformed to synthetic LSST system',
                               units='nJy')
        monsterSchema.addField(fluxcolname_err, type='D',
                               doc='error on flux transformed to synthetic LSST system',
                               units='nJy')
        monsterSchema.addField(flagcolname, type='I',
                               doc='source of flux (0:VST, 1:Skymapper, 2:PS1, 3:GaiaXP, 4:DES)',
                               units='')

    return monsterSchema

def makeMonsterCat(monsterSchema, monsterTable):
    """
    Make the Gaia catalog with transformed and rank-ordered reference fluxes
    to persist.

    Parameters
    ----------
    monsterSchema: `lsst.afw.table.Schema`
       Monster reference catalog schema
    monsterTable: `Astropy Table`
       Monster reference catalog to convert

    Returns
    -------
    monsterCat: `lsst.afw.table.BaseCatalog`
       Monster reference catalog for persistence
    """

    monsterCat = afwTable.SimpleCatalog(monsterSchema)
    monsterCat.resize(np.size(monsterTable))
    for col in monsterTable.itercols():
        monsterCat[col.name] = monsterTable[col.name]

    return monsterCat