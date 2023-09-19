import os
from astropy import units
import numpy as np

from lsst.afw.table import SimpleCatalog
import lsst.afw.table as afwTable
import lsst.geom as geom

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

def getSurveyFluxColumnName(survey, band):
    """
    Retrieve the flux column names for a given survey.

    Parameters
    ----------
    survey : `str`
        Name of the survey whose refcat we are writing
    bands : `List` of `str`
        Names of the bands in the refcat

    Returns
    -------
    fluxname: `str`
    """

    if survey == 'GaiaXP':
        fluxname = 'Decam_flux_'+band+'_flux'
    elif survey == 'PS1':
        fluxname = band+'_flux'
    elif survey == 'SkyMapper':
        fluxname = band+'_psf_flux'
    elif survey == 'VST':
        fluxname = band.upper()+'APERMAG3_flux'
    else:
        print('Could not find '+survey+' survey for flux column formatting.')

    return fluxname

def makeRefSchema(survey, bands):
    """
    Make the refcat schema

    Parameters
    ----------
    survey : `str`
        Name of the survey whose refcat we are writing
    bands : `List` of `str`
        Names of the bands in the refcat

    Returns
    -------
    refSchema: `lsst.afw.table.Schema`
    """

    refSchema = afwTable.SimpleTable.makeMinimalSchema()
    refSchema.addField('GaiaDR3_id', type='L', doc='Gaia DR3 ID')

    for band in bands:
        colname = 'decam_'+band+'_from_'+survey+'_flux'
        colname_err = colname+'Err'
        refSchema.addField(colname, type='D',
                           doc='flux transformed to DECam system')
        refSchema.addField(colname_err, type='D',
                           doc='error on flux transformed to DECam system')

        # fluxname = getSurveyFluxColumnName(survey, band)
        # fluxerrname = fluxname+'Err'
        # refSchema.addField(fluxname, type='D',
        #                    doc='flux in original system')
        # refSchema.addField(fluxerrname, type='D',
        #                    doc='error on flux in original system')

    return refSchema

def makeRefCat(refSchema, refTable, survey, bands):
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

    Returns
    -------
    refCat: `lsst.afw.table.BaseCatalog`
       Standard star catalog for persistence
    """

    refCat = afwTable.SimpleCatalog(refSchema)
    refCat.resize(np.size(refTable))

    refCat['id'][:] = refTable['id']
    refCat['GaiaDR3_id'][:] = refTable['GaiaDR3_id']
    refCat['coord_ra'][:] = refTable['coord_ra'] * geom.degrees
    refCat['coord_dec'][:] = refTable['coord_dec'] * geom.degrees
    
    for band in bands:
        colname = 'decam_'+band+'_from_'+survey+'_flux'
        colname_err = colname+'Err'
        # import pdb; pdb.set_trace()
        refCat[colname][:] = refTable[colname]
        refCat[colname_err][:] = refTable[colname_err]
 
        # fluxname = getSurveyFluxColumnName(survey, band)
        # fluxerrname = fluxname+'Err'
        # refCat[fluxname][:] = refTable[fluxname]
        # refCat[fluxerrname][:] = refTable[fluxerrname]

    return refCat
