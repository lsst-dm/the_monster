import os
import numpy as np
from astropy import units
import warnings
import lsst.utils

from abc import ABC, abstractmethod


__all__ = [
    "FLAG_DICT",
    "DESInfo",
    "GaiaDR3Info",
    "GaiaXPInfo",
    "GaiaXPuInfo",
    "LATISSInfo",
    "PS1Info",
    "SDSSInfo",
    "SDSSuInfo",
    "SkyMapperInfo",
    "SynthLSSTInfo",
    "VSTInfo",
    "ComCamInfo",
    "MonsterInfo",
]

FLAG_DICT = {
    "GaiaDR3": 0,
    "DES": 2,
    "GaiaXP": 4,
    "PS1": 8,
    "SkyMapper": 16,
    "VST": 32,
    "SDSS": 64,
    "SLR": 128,
    "SynthLSST": 256,
    "LATISS": 512,
    "Monster": 1024,
    "ComCam": 2048,
}


class RefcatInfo(ABC):
    PATH = ""
    TRANSFORMED_PATH = None
    NAME = ""
    COLORTERM_PATH = None

    def __init__(self, path=None, transformed_path=None, name=None, flag=None):
        if path is None:
            self._path = self.PATH
        else:
            self._path = path

        if transformed_path is None:
            if self.TRANSFORMED_PATH is None:
                self._transformed_path = self._path + "_transformed"
            else:
                self._transformed_path = self.TRANSFORMED_PATH
        else:
            self._transformed_path = transformed_path

        if self.COLORTERM_PATH is None:
            self._colorterm_path = os.path.join(
                lsst.utils.getPackageDir("the_monster"),
                "colorterms",
            )
        else:
            self._colorterm_path = self.COLORTERM_PATH

        if name is None:
            self._name = self.NAME
        else:
            self._name = name

        if flag is None:
            self._flag = self.FLAG
        else:
            self._flag = flag

    @property
    def path(self):
        return self._path

    @property
    def transformed_path(self):
        return self._transformed_path

    @property
    def flag(self):
        return self._flag

    def colorterm_file(self, band):
        """Get the colorterm correction file for this band/catalog.

        Parameters
        ----------
        band : `str`

        Returns
        -------
        filename : `str`
        """
        filename = os.path.join(
            self._colorterm_path,
            f"{self.name}_to_DES_band_{band}.yaml",
        )

        return filename

    def uband_offset_file(self, target_name):
        """Get the uband SLR offset map file for the given target survey.

        Parameters
        ----------
        target_name : `str`
            Name of the target survey.

        Returns
        -------
        filename : `str`
        """
        filename = os.path.join(
            self._colorterm_path,
            "offsets",
            f"uSLR_to_{target_name}_offset_map.hsp",
        )

        return filename

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

    def get_gmi_color_range(self):
        """Get the color range appropriate for the g-i color.

        Returns
        -------
        color_low, color_high : `float`
        """
        return (-100.0, 100.0)

    def get_imz_color_range(self):
        """Get the color range appropriate for the i-z color.

        Returns
        -------
        color_low, color_high : `float`
        """
        return (-100.0, 100.0)

    def get_gmr_color_range(self):
        """Get the color range appropriate for the g-r color.

        Returns
        -------
        color_low, color_high : `float`
        """
        return (-100.0, 100.0)

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
        elif band in ["u"]:
            color_range = self.get_gmr_color_range()

        return color_range

    def get_mag_range(self, band):
        """Get the appropriate magnitude range for a given band.

        Parameters
        ----------
        band : `str`
            Band to get magnitude range.

        Returns
        -------
        mag_low, mag_high : `float`
        """
        return (-np.inf, np.inf)

    def get_sn_range(self, band):
        """Get the appropriate signal-to-noise range for a given band.

        Parameters
        ----------
        band : `str`
            Band to get signal-to-noise range.

        Returns
        -------
        sn_low, sn_high : `float`
        """
        return (0.0, np.inf)

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
        elif band in ["u"]:
            band_1 = "g"
            band_2 = "r"
        else:
            raise NotImplementedError("Unsupported band: ", band)

        return band_1, band_2

    def get_slr_band(self, band):
        """Get the appropriate band for SLR measurements.

        Parameters
        ----------
        band : `str`
            Band to get SLR corrections from.

        Returns
        -------
        slr_band : `str`
        """
        if band == "u":
            return "g"
        else:
            return band

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

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mag_color_1 = (np.array(catalog[flux_color_field_1])*units.nJy).to_value(units.ABmag)
            mag_color_2 = (np.array(catalog[flux_color_field_2])*units.nJy).to_value(units.ABmag)

        mag_color = mag_color_1 - mag_color_2

        return mag_color

    def get_transformed_flux_field(self, band):
        """Get the transformed-to-internal reference flux field associated
           with a band. This should be decam for grizy-bands and sdss
           for u-band
        Parameters
        ----------
        band : `str`
            Name of band to get flux field.

        Returns
        -------
        flux_field : `str`
            Name of flux field appropriate for this catalog.
        """
        if band in "grizy":
            return f"decam_{band}_from_{self.NAME}_flux"
        elif band in "u":
            return f"sdss_{band}_from_{self.NAME}_flux"
        else:
            raise ValueError(f"Unsupported band: {band}")

    def get_transformed_mag_colors(self, catalog, band):
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

        flux_color_field_1 = self.get_transformed_flux_field(band_1)
        flux_color_field_2 = self.get_transformed_flux_field(band_2)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
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
        mag_range = self.get_mag_range(band)
        sn_range = self.get_sn_range(band)

        flux = np.array(catalog[flux_field])
        flux_err = np.array(catalog[flux_field + "Err"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)

            sn = flux / flux_err
            mag = (flux*units.nJy).to_value(units.ABmag)

            selected = (
                (mag_color > color_range[0])
                & (mag_color < color_range[1])
                & (np.isfinite(mag_color))
                & (np.isfinite(flux))
                & (np.isfinite(sn))
                & (np.isfinite(mag))
                & (flux < 1e10)
                & (mag > mag_range[0])
                & (mag < mag_range[1])
                & (sn > sn_range[0])
                & (sn < sn_range[1])
            )

        return selected


class GaiaDR3Info(RefcatInfo):
    PATH = "/sdf/data/rubin/shared/the_monster/GAIA_DR3/gaia_dr3"
    NAME = "GaiaDR3"
    FLAG = FLAG_DICT[NAME]

    def get_flux_field(self, band):
        return f"phot_{band.lower()}_mean_flux"

    def get_gmi_color_range(self):
        return (0.5, 3.0)

    def get_imz_color_range(self):
        return (0.0, 0.7)


class GaiaXPInfo(RefcatInfo):
    PATH = "/sdf/data/rubin/shared/the_monster/sharded_refcats/gaia_xp_ps_des_sdss_sm_20240116"
    NAME = "GaiaXP"
    FLAG = FLAG_DICT[NAME]
    bands = ["g", "r", "i", "z", "y"]

    def get_flux_field(self, band):
        _band = band
        if band == "y":
            _band = "Y"
        return f"Decam_flux_{_band}_flux"

    def get_gmi_color_range(self):
        return (0.3, 3.0)

    def get_imz_color_range(self):
        return (0.0, 0.7)

    def get_mag_range(self, band):
        if band == "g":
            return (-np.inf, 19.3)
        elif band == "r":
            return (-np.inf, 18.0)
        elif band == "i":
            return (-np.inf, 17.5)
        elif band == "z":
            return (-np.inf, 17.5)
        elif band == "y":
            return (-np.inf, 17.4)
        else:
            return (-np.inf, np.inf)


class DESInfo(RefcatInfo):
    PATH = "/sdf/data/rubin/shared/the_monster/sharded_refcats/des_y6_calibration_stars_20230511"
    NAME = "DES"
    FLAG = FLAG_DICT[NAME]
    bands = ["g", "r", "i", "z", "y"]

    def get_flux_field(self, band):
        _band = band
        # We use the g band in place of the u band (for SLR).
        if _band == "u":
            _band = "g"

        return f"MAG_STD_{_band.upper()}_flux"

    def get_gmi_color_range(self):
        return (0.0, 3.5)

    def get_imz_color_range(self):
        return (0.0, 0.8)

    def get_gmr_color_range(self):
        # This is used for u-band SLR calibrations.
        return (0.23, 0.7)

    def get_mag_range(self, band):
        if band == "g":
            return (16.25, np.inf)
        elif band == "r":
            return (16.0, np.inf)
        elif band == "i":
            return (15.5, np.inf)
        elif band == "z":
            return (15.0, np.inf)
        elif band == "y":
            return (15.0, np.inf)
        else:
            return (-np.inf, np.inf)

    def get_sn_range(self, band):
        return (10.0, np.inf)

    def select_stars(self, catalog, band):
        selected = super().select_stars(catalog, band)

        selected &= (
            (catalog["NGOOD_G"] > 2)
            & (catalog["NGOOD_R"] > 2)
            & (catalog["NGOOD_I"] > 2)
            & (catalog["NGOOD_Z"] > 2)
            & ((np.array(catalog["MAG_STD_R_flux"])*units.nJy).to_value(units.ABmag) > 16.0)
        )

        if band == "y":
            selected &= (catalog["NGOOD_Y"] > 2)

        return selected


class SkyMapperInfo(RefcatInfo):
    PATH = "/sdf/data/rubin/shared/the_monster/sharded_refcats/sky_mapper_dr2_20221205"
    NAME = "SkyMapper"
    FLAG = FLAG_DICT[NAME]
    bands = ["g", "r", "i", "z"]

    def get_flux_field(self, band):
        return f"{band}_psf_flux"

    def get_gmi_color_range(self):
        return (0.2, 2.7)

    def get_imz_color_range(self):
        return (0.0, 0.7)

    def get_mag_range(self, band):
        if band == "g":
            return (13.0, 21.0)
        elif band == "r":
            return (12.5, 21.0)
        elif band == "i":
            return (12.0, 20.6)
        elif band == "z":
            return (12.0, 19.75)
        else:
            return (-np.inf, np.inf)

    def get_sn_range(self, band):
        if band == "g":
            return (50.0, np.inf)
        elif band == "r":
            return (50.0, np.inf)
        elif band == "i":
            return (50.0, np.inf)
        elif band == "z":
            return (50.0, np.inf)
        else:
            return (0.0, np.inf)


class PS1Info(RefcatInfo):
    PATH = "/sdf/group/rubin/datasets/refcats/htm/v1/ps1_pv3_3pi_20170110"
    TRANSFORMED_PATH = "/sdf/data/rubin/shared/the_monster/sharded_refcats/ps1_transformed"
    NAME = "PS1"
    FLAG = FLAG_DICT[NAME]
    bands = ["g", "r", "i", "z", "y"]

    def get_flux_field(self, band):
        return f"{band}_flux"

    def get_gmi_color_range(self):
        return (0.3, 2.8)

    def get_imz_color_range(self):
        return (0.0, 0.7)

    def get_mag_range(self, band):
        if band == "g":
            return (13.0, 21.25)
        elif band == "r":
            return (13.25, 20.25)
        elif band == "i":
            return (13.75, 19.5)
        elif band == "z":
            return (13.75, 19.0)
        elif band == "y":
            return (13.75, 17.0)
        else:
            return (-np.inf, np.inf)

    def get_sn_range(self, band):
        if band == "g":
            return (50.0, np.inf)
        elif band == "r":
            return (30.0, np.inf)
        elif band == "i":
            return (30.0, np.inf)
        elif band == "z":
            return (30.0, np.inf)
        elif band == "y":
            return (30.0, np.inf)
        else:
            return (0.0, np.inf)

    def select_stars(self, catalog, band):
        selected = super().select_stars(catalog, band)

        unique_ids, unique_index = np.unique(catalog["id"], return_index=True)
        if len(unique_ids) < len(catalog):
            unique_selected = np.zeros(len(catalog), dtype=bool)
            unique_selected[unique_index] = True
            selected &= unique_selected

        return selected


class VSTInfo(RefcatInfo):
    PATH = "/sdf/data/rubin/shared/the_monster/sharded_refcats/vst_atlas_20221205"
    NAME = "VST"
    FLAG = FLAG_DICT[NAME]
    bands = ["g", "r", "i", "z"]

    def get_flux_field(self, band):
        return f"{band.upper()}APERMAG3_flux"

    def get_gmi_color_range(self):
        return (0.5, 3.1)

    def get_imz_color_range(self):
        return (0.2, 0.8)

    def get_mag_range(self, band):
        if band == "g":
            return (18.0, 21.5)
        elif band == "r":
            return (17.0, 21.0)
        elif band == "i":
            return (16.0, 20.5)
        elif band == "z":
            return (15.5, 19.25)
        else:
            return (-np.inf, np.inf)

    def get_sn_range(self, band):
        if band == "g":
            return (20.0, np.inf)
        elif band == "r":
            return (20.0, np.inf)
        elif band == "i":
            return (20.0, np.inf)
        elif band == "z":
            return (20.0, np.inf)
        else:
            return (0.0, np.inf)


class SynthLSSTInfo(RefcatInfo):
    NAME = "SynthLSST"
    FLAG = FLAG_DICT[NAME]
    bands = ["u", "g", "r", "i", "z", "y"]

    def get_flux_field(self, band):
        return f"{band}_flux"

    # These color ranges are determined by the template stars.
    def get_gmi_color_range(self):
        return (0.3, 3.0)

    def get_imz_color_range(self):
        return (0.0, 0.6)


class GaiaXPuInfo(GaiaXPInfo):
    NAME = "GaiaXPu"
    bands = ["u"]
    TRANSFORMED_PATH = "/sdf/data/rubin/shared/the_monster/sharded_refcats/gaia_xp_u_20240116_transformed"

    def get_flux_field(self, band):
        return f"Sdss_flux_{band}_flux"

    # This is used for the u-band calibration.
    def get_gmr_color_range(self):
        return (0.25, 0.8)

    def get_sn_range(self, band):
        return (10.0, np.inf)

    def colorterm_file(self, band):
        filename = os.path.join(
            self._colorterm_path,
            f"{self.name}_to_SDSS_band_{band}.yaml",
        )

        return filename


class SDSSInfo(RefcatInfo):
    PATH = "/sdf/data/rubin/shared/the_monster/sharded_refcats/sdss_16_standards_20221205"
    NAME = "SDSS"
    FLAG = FLAG_DICT[NAME]
    bands = ["u", "g", "r", "i", "z"]

    def get_flux_field(self, band):
        return f"psfMag_{band}_flux"

    # This is used for the u-band calibration.
    def get_gmr_color_range(self):
        return (0.25, 0.8)

    def get_sn_range(self, band):
        return (10.0, np.inf)

    def get_mag_range(self, band):
        if band == "u":
            return (15.0, 21.5)
        else:
            return super().get_mag_range(band)

    def colorterm_file(self, band):
        if band == "u":
            # This is not transformed.
            filename = os.path.join(
                self._colorterm_path,
                f"{self.name}_to_SDSS_band_{band}.yaml",
            )
        else:
            filename = super().colorterm_file(band)

        return filename


class SDSSuInfo(SDSSInfo):
    bands = ["u"]


class LATISSInfo(RefcatInfo):
    NAME = "LATISS"
    FLAG = FLAG_DICT[NAME]
    bands = ["g", "r", "i", "z", "y"]

    def get_flux_field(self, band):
        return f"{band}_flux"

    def get_gmi_color_range(self):
        return (0.5, 2.0)

    def get_imz_color_range(self):
        return (0.05, 0.6)


class MonsterInfo(RefcatInfo):
    PATH = "/sdf/data/rubin/shared/refcats/the_monster_20240904"
    NAME = "Monster"
    FLAG = FLAG_DICT[NAME]
    bands = ["u", "g", "r", "i", "z", "y"]

    def get_flux_field(self, band):
        if band == "u":
            return f"monster_SDSS_{band}_flux"
        else:
            return f"monster_DES_{band}_flux"

    def get_gmi_color_range(self):
        return (0.0, 3.5)

    def get_imz_color_range(self):
        return (0.0, 0.75)


class ComCamInfo(RefcatInfo):
    NAME = "ComCam"
    FLAG = FLAG_DICT[NAME]
    bands = ["u", "g", "r", "i", "z", "y"]

    def get_flux_field(self, band):
        # Need name after conversion.
        return f"comcam_{band}_flux"

    def get_gmr_color_range(self):
        return (0.35, 0.7)

    def get_gmi_color_range(self):
        return (0.0, 3.5)

    def get_imz_color_range(self):
        return (0.0, 0.75)
