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

    # This is to patch test files.
    if "TestGaiaDR3_id" in stars.columns:
        stars.rename_column("TestGaiaDR3_id", "GaiaDR3_id")

    return stars


def makeRefSchema(cat_info, reference_name):
    """
    Make the refcat schema

    Parameters
    ----------
    cat_info : `RefcatInfo`
        Reference catalog information for the
        refcat we are writing
    reference_name : `str`
        Name of the overall reference catalog

    Returns
    -------
    refSchema: `lsst.afw.table.Schema`
    """

    refSchema = afwTable.SimpleTable.makeMinimalSchema()
    refSchema.addField(reference_name+'_id', type='L', doc='Gaia DR3 ID')

    for band in cat_info.bands:
        # use cat info class to get transformed flux field name
        colname = cat_info.get_transformed_flux_field(band)
        colname_err = colname + 'Err'

        # check which internal bandpass is used
        # this should be decam for everything except for u band
        if 'decam_' in colname:
            target_system = 'DECam'
        elif 'sdss_' in colname:
            target_system = 'SDSS'
        else:
            raise ValueError(f"Unknown target system for band {band}")

        refSchema.addField(colname, type='D',
                           doc=f'flux transformed to {target_system} system',
                           units='nJy')
        refSchema.addField(colname_err, type='D',
                           doc=f'error on flux transformed to {target_system} system',
                           units='nJy')

    return refSchema


def makeRefCat(refSchema, refTable, cat_info, reference_name):
    """
    Make the standard star catalog for persistence

    Parameters
    ----------
    refSchema: `lsst.afw.table.Schema`
       Standard star catalog schema
    refTable: `Astropy Table`
       Reference catalog to convert
    cat_info : `RefcatInfo`
        Reference catalog information for the
        refcat we are writing
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

    for band in cat_info.bands:
        colname = cat_info.get_transformed_flux_field(band)
        colname_err = colname + 'Err'
        refCat[colname][:] = refTable[colname]
        refCat[colname_err][:] = refTable[colname_err]

    return refCat


def makeMonsterSchema(gaia_catalog_columns, bands, output_system='lsst'):
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
    output_system : `str`
        Name of the output system to use.

    Returns
    -------
    monsterSchema: `lsst.afw.table.Schema`
    """

    monsterSchema = afwTable.SimpleTable.makeMinimalSchema()

    # We want to transfer all the existing Gaia columns to the output table,
    # as well as the newly-added fluxes, errors, and flags. The "transfer" loop
    # below takes care of adding the Gaia columns to the schema.
    # Columns "id", "coord_ra", and "coord_dec" already exist in the minimal
    # schema that we have initialized, so we don't need to include them in the
    # loop to transfer entries. Also, the flux/error/flag columns don't have
    # units in the input table, so we'll treat them separately to add the
    # units. Thus we exclude them from the "transfer" loop as well.
    exclude_columns = ["id", "coord_ra", "coord_dec"]
    for band in bands:
        exclude_columns.append(f"monster_{output_system}_{band}_flux")
        exclude_columns.append(f"monster_{output_system}_{band}_fluxErr")
        exclude_columns.append(f"monster_{output_system}_{band}_source_flag")

    fieldtype_dict = {'float32': 'F', 'float64': 'D',
                      'int64': 'L', 'bool': 'B'}

    # Transfer the existing Gaia columns to the new table schema.
    for col in gaia_catalog_columns:
        # Skip the "exclude_columns," which will get treated separately.
        if col.name not in exclude_columns:
            # If the Gaia input column had units, use those:
            if col.unit is not None:
                monsterSchema.addField(col.name,
                                       type=fieldtype_dict[col.dtype.name],
                                       doc=col.description,
                                       units=col.unit.to_string()
                                       )
            # Otherwise, initialize a unitless column.
            else:
                monsterSchema.addField(col.name,
                                       type=fieldtype_dict[col.dtype.name],
                                       doc=col.description,
                                       units=''
                                       )

    # Add columns to the schema for the flux, flux error, and flags.
    for band in bands:
        fluxcolname = f"monster_{output_system}_{band}_flux"
        fluxcolname_err = fluxcolname+'Err'
        flagcolname = f"monster_{output_system}_{band}_source_flag"
        monsterSchema.addField(fluxcolname, type='D',
                               doc='flux transformed to synthetic system',
                               units='nJy')
        monsterSchema.addField(fluxcolname_err, type='D',
                               doc='error on flux transformed to synthetic system',
                               units='nJy')
        monsterSchema.addField(flagcolname, type='I',
                               doc='source of flux (1:VST, 2:Skymapper, 4:PS1, 8:GaiaXP, 16:DES)',
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
    # Convert RA, Dec to radians for the output afwTable
    monsterCat['coord_ra'][:] = np.deg2rad(monsterTable['coord_ra'])
    monsterCat['coord_dec'][:] = np.deg2rad(monsterTable['coord_dec'])

    return monsterCat
