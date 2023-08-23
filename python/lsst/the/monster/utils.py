import os
from astropy import units

from lsst.afw.table import SimpleCatalog


__all__ = ["read_stars"]


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
