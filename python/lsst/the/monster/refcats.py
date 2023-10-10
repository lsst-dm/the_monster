import numpy as np
from astropy import units

from abc import ABC, abstractmethod


__all__ = ["GaiaDR3Info", "GaiaXPInfo", "DESInfo", "SkyMapperInfo", "PS1Info", "VSTInfo"]


class RefcatInfo(ABC):
    PATH = ""
    NAME = ""

    def __init__(self, path=None, name=None):
        if path is None:
            self._path = self.PATH
        else:
            self._path = path

        if name is None:
            self._name = self.NAME
        else:
            self._name = name

    @property
    def path(self):
        return self._path

    @property
    def name(self):
        return self._name

    @abstractmethod
    def get_flux_field(self, band):
        """Get the flux field associated with a band.

        Parameters
        ----------
        band : `str`
            Name of band to get flux field.

        Returns
        -------
        flux_field : `str`
            Name of flux field appropriate for this catalog.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_gmi_color_range(self):
        """Get the color range appropriate for the g-i color.

        Returns
        -------
        color_low, color_high : `float`
        """
        raise NotImplementedError()

    @abstractmethod
    def get_imz_color_range(self):
        """Get the color range appropriate for the i-z color.

        Returns
        -------
        color_low, color_high : `float`
        """
        raise NotImplementedError()

    def get_color_range(self, band):
        """Get the appropriate color range for a given band.

        Parameters
        ----------
        band : `str`
            Band to correct.

        Returns
        -------
        color_low, color_high : `float`
        """
        if band in ["g", "r", "i"]:
            color_range = self.get_gmi_color_range()
        elif band in ["z", "y"]:
            color_range = self.get_imz_color_range()

        return color_range

    def get_color_bands(self, band):
        """Get the appropriate bands to compute a color to correct
        a given band.

        Parameters
        ----------
        band : `str`
            Band to correct.

        Returns
        -------
        band_1, band_2 : `str`
        """
        if band in ["g", "r", "i"]:
            band_1 = "g"
            band_2 = "i"
        elif band in ["z", "y"]:
            band_1 = "i"
            band_2 = "z"
        else:
            raise NotImplementedError("Unsupported band: ", band)

        return band_1, band_2

    def get_mag_colors(self, catalog, band):
        """Get magnitude colors appropriate for correcting a given band.

        Parameters
        ----------
        catalog : `lsst.afw.table.SimpleCatalog`
            Input catalog.
        band : `str`
            Name of band for doing selection.

        Returns
        -------
        colors : `np.ndarray`
            Array of colors used for color terms for given band.
        """
        band_1, band_2 = self.get_color_bands(band)

        flux_color_field_1 = self.get_flux_field(band_1)
        flux_color_field_2 = self.get_flux_field(band_2)

        mag_color_1 = (np.array(catalog[flux_color_field_1])*units.nJy).to_value(units.ABmag)
        mag_color_2 = (np.array(catalog[flux_color_field_2])*units.nJy).to_value(units.ABmag)
        mag_color = mag_color_1 - mag_color_2

        return mag_color

    def select_stars(self, catalog, band):
        """Star selection appropriate for this type of refcat.

        Parameters
        ----------
        catalog : `lsst.afw.table.SimpleCatalog`
            Input catalog.
        band : `str`
            Name of band for doing selection.

        Returns
        -------
        selected : `np.ndarray`
            Boolean array of selected (True) or not (False).
        """
        mag_color = self.get_mag_colors(catalog, band)
        flux_field = self.get_flux_field(band)
        color_range = self.get_color_range(band)

        flux = np.array(catalog[flux_field])

        selected = (
            (mag_color > color_range[0])
            & (mag_color < color_range[1])
            & (np.isfinite(mag_color))
            & (np.isfinite(flux))
            & (flux < 1e10)
        )

        return selected


class GaiaDR3Info(RefcatInfo):
    PATH = "/sdf/data/rubin/shared/the_monster/GAIA_DR3/gaia_dr3"
    NAME = "GaiaDR3"

    def get_flux_field(self, band):
        return f"phot_{band.lower()}_mean_flux"

    def get_gmi_color_range(self):
        return (0.5, 3.0)

    def get_imz_color_range(self):
        return (0.0, 0.7)


class GaiaXPInfo(RefcatInfo):
    PATH = "/sdf/data/rubin/shared/the_monster/sharded_refcats/gaia_xp_ps_des_sdss_sm_20221216"
    NAME = "GaiaXP"
    bands = ['g', 'r', 'i', 'z', 'y']

    def get_flux_field(self, band):
        _band = band
        if band == "y":
            _band = "Y"
        return f"Decam_flux_{_band}_flux"

    def get_gmi_color_range(self):
        return (0.5, 3.0)

    def get_imz_color_range(self):
        return (0.0, 0.7)


class DESInfo(RefcatInfo):
    PATH = "/sdf/data/rubin/shared/the_monster/sharded_refcats/des_y6_calibration_stars_20230511"
    NAME = "DES"
    bands = ['g', 'r', 'i', 'z', 'y']

    def get_flux_field(self, band):
        return f"MAG_STD_{band.upper()}_flux"

    def get_gmi_color_range(self):
        return (0.5, 3.5)

    def get_imz_color_range(self):
        return (0.0, 0.8)

    def select_stars(self, catalog, band):
        selected = super().select_stars(catalog, band)

        selected &= (
            (catalog["NGOOD_G"] > 2)
            & (catalog["NGOOD_R"] > 2)
            & (catalog["NGOOD_I"] > 2)
            & (catalog["NGOOD_Z"] > 2)
        )

        if band == "y":
            selected &= (catalog["NGOOD_Y"] > 2)

        return selected


class SkyMapperInfo(RefcatInfo):
    PATH = "/sdf/data/rubin/shared/the_monster/sharded_refcats/sky_mapper_dr2_20221205"
    NAME = "SkyMapper"
    bands = ['g', 'r', 'i', 'z']

    def get_flux_field(self, band):
        return f"{band}_psf_flux"

    def get_gmi_color_range(self):
        return (0.5, 2.7)

    def get_imz_color_range(self):
        return (0.0, 0.7)


class PS1Info(RefcatInfo):
    PATH = "/fs/ddn/sdf/group/rubin/ncsa-datasets/refcats/htm/v1/ps1_pv3_3pi_20170110"
    NAME = "PS1"
    bands = ['g', 'r', 'i', 'z']

    def get_flux_field(self, band):
        return f"{band}_flux"

    def get_gmi_color_range(self):
        return (0.5, 2.8)

    def get_imz_color_range(self):
        return (0.0, 0.7)


class VSTInfo(RefcatInfo):
    PATH = "/sdf/data/rubin/shared/the_monster/sharded_refcats/vst_atlas_20221205"
    NAME = "VST"
    bands = ['g', 'r', 'i', 'z']

    def get_flux_field(self, band):
        return f"{band.upper()}APERMAG3_flux"

    def get_gmi_color_range(self):
        return (0.5, 3.1)

    def get_imz_color_range(self):
        return (0.2, 0.8)
