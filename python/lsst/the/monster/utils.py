import os
from astropy import units
import numpy as np

from lsst.afw.table import SimpleCatalog
import lsst.afw.table as afwTable

__all__ = ["read_stars", "makeRefSchema", "makeRefCat"]


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
        try:
            temp = SimpleCatalog.readFits(os.path.join(path, str(index) + ".fits"))
        except RuntimeError as e:
            if allow_missing:
                continue
            else:
                raise e
        if stars is None:
            stars = temp
        else:
            stars.extend(temp)

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
