import numpy as np
from astropy import units
from smatch import Matcher
import esutil
import matplotlib.pyplot as plt
import warnings

import lsst.sphgeom as sphgeom

from .refcats import GaiaDR3Info, GaiaXPInfo, GaiaXPuInfo, DESInfo, SkyMapperInfo, PS1Info, VSTInfo, SDSSInfo
from .splinecolorterms import ColortermSplineFitter, ColortermSpline, MagSplineFitter
from .utils import read_stars
from .refcats import RefcatInfo


__all__ = [
    "read_uband_combined_catalog",
    "UBandSLRSplineMeasurer",
]


class _CombinedInfo(RefcatInfo):
    FLAG = -1

    def get_flux_field(self, band):
        if band == "u":
            return "ref_u_flux"
        else:
            return f"{band}_flux"


def read_uband_combined_catalog(
    gaia_ref_info,
    cat_info_list,
    uband_ref_info,
    htm_pixel_list,
    testing_mode=False,
):
    """Read the uband combined catalog for a set of HTM indices.

    This will combine all the catalogs according to proper ranking, and
    apply all relevant color terms to put them in the intermediate (DES)
    system. It will also do any corrections to the reference u-band
    flux that are necessary.

    Parameters
    ----------
    gaia_ref_info : `RefcatInfo`
        Info for the astrometric (Gaia) reference.
    cat_info_list : `list` [`RefcatInfo`]
        Reverse priority list of refcat infos to load.
    uband_ref_info : `RefcatInfo`
        Info for the u-band reference catalog.
    htm_pixel_list : `list` [`int`]
        List of HTM pixel ids to load.
    testing_mode : `bool`, optional
        Running in "testing" mode?

    Returns
    -------
    gaia_stars_all : `astropy.table.Table`
        Table of stars, with fluxes filled in.
    """
    u_slr_bands = uband_ref_info.get_color_bands("u")

    # Read these in first, to "fail" fast.
    uband_ref_stars = read_stars(uband_ref_info.path, htm_pixel_list, allow_missing=True)
    if len(uband_ref_stars) == 0:
        return []

    gaia_stars_all = read_stars(gaia_ref_info.path, htm_pixel_list, allow_missing=testing_mode)

    if len(gaia_stars_all) == 0:
        return []

    nan_column = np.full(len(gaia_stars_all), np.nan)
    # We need to add in columns for flux/err for u, g, r.
    for band in u_slr_bands:
        gaia_stars_all.add_column(nan_column, name=f"{band}_flux")
        gaia_stars_all.add_column(nan_column, name=f"{band}_fluxErr")

    gaia_stars_all.add_column(nan_column, name="ref_u_flux")
    gaia_stars_all.add_column(nan_column, name="ref_u_fluxErr")

    # Loop over the refcats.
    for cat_info in cat_info_list:
        # We read from the "transformed_path" which has the (DES)
        # transformed catalog.
        cat_stars = read_stars(cat_info.transformed_path, htm_pixel_list, allow_missing=True)
        if len(cat_stars) == 0:
            continue

        if testing_mode:
            # We need to hack the catalogs for the test names (sorry).
            for name in cat_stars.dtype.names:
                if (substr := "_" + cat_info.ORIG_NAME_FOR_TEST + "_") in name:
                    new_name = name.replace(substr, "_" + cat_info.NAME + "_")
                    cat_stars.rename_column(name, new_name)

        for band in u_slr_bands:
            cat_flux_field = cat_info.get_transformed_flux_field(band)
            selected = np.isfinite(cat_stars[cat_flux_field])

            if selected.sum() == 0:
                continue

            cat_stars_selected = cat_stars[selected]

            a, b = esutil.numpy_util.match(gaia_stars_all["id"],
                                           cat_stars_selected["GaiaDR3_id"])
            gaia_stars_all[f"{band}_flux"][a] = cat_stars_selected[cat_flux_field][b]
            gaia_stars_all[f"{band}_fluxErr"][a] = cat_stars_selected[cat_flux_field + "Err"][b]

    # Work with the reference stars.
    ref_colorterm_spline = ColortermSpline.load(uband_ref_info.colorterm_file("u"))

    band_1, band_2 = uband_ref_info.get_color_bands("u")
    uband_orig_flux = uband_ref_stars[uband_ref_info.get_flux_field("u")]
    uband_orig_flux_err = uband_ref_stars[uband_ref_info.get_flux_field("u") + "Err"]
    uband_model_flux = ref_colorterm_spline.apply(
        uband_ref_stars[uband_ref_info.get_flux_field(band_1)],
        uband_ref_stars[uband_ref_info.get_flux_field(band_2)],
        uband_orig_flux,
    )

    uband_ref_selected = uband_ref_info.select_stars(uband_ref_stars, "u")
    uband_ref_selected &= np.isfinite(uband_model_flux)

    uband_ref_stars = uband_ref_stars[uband_ref_selected]
    uband_orig_flux = uband_orig_flux[uband_ref_selected]
    uband_orig_flux_err = uband_orig_flux_err[uband_ref_selected]
    uband_model_flux = uband_model_flux[uband_ref_selected]

    with Matcher(gaia_stars_all["coord_ra"], gaia_stars_all["coord_dec"]) as m:
        idx, i1, i2, d = m.query_knn(
            uband_ref_stars["coord_ra"],
            uband_ref_stars["coord_dec"],
            distance_upper_bound=0.5/3600.,
            return_indices=True,
        )

    gaia_stars_all["ref_u_flux"][i1] = uband_model_flux[i2]
    gaia_stars_all["ref_u_fluxErr"][i1] = uband_model_flux[i2] * (
        uband_orig_flux_err[i2]/uband_orig_flux[i2]
    )

    return gaia_stars_all


class UBandSLRSplineMeasurer:
    def __init__(
        self,
        gaia_reference_class=GaiaDR3Info,
        catalog_info_class_list=[VSTInfo, SkyMapperInfo,
                                 PS1Info, GaiaXPInfo, DESInfo],
        uband_ref_class=SDSSInfo,
        uband_slr_class=DESInfo,
        do_fit_flux_offset=False,
        do_fit_mag_offsets=True,
        testing_mode=False,
        htm_level=7,
    ):
        self.gaia_reference_class = gaia_reference_class
        self.catalog_info_class_list = catalog_info_class_list
        self.uband_ref_class = uband_ref_class
        self.uband_slr_class = uband_slr_class
        self.do_fit_flux_offset = do_fit_flux_offset
        self.do_fit_mag_offsets = do_fit_mag_offsets
        self.testing_mode = testing_mode
        self.htm_level = 7

    @property
    def ra_dec_range(self):
        """Get ra/dec range to do fit.

        Returns
        -------
        ra_min, ra_max, dec_min, dec_max : `float`
        """
        return (-10.0, 10.0, -20.0, 10.0)

    @property
    def n_nodes(self):
        return 7

    @property
    def n_mag_nodes(self):
        return 7

    def measure_uband_slr_spline_fit(self, do_plots=True, overwrite=False):
        """Measure the u-band SLR spline fit.

        Parameters
        ----------
        do_plots : `bool`, optional
            Make QA plots.
        overwrite : `bool`, optional
            Overwrite an existing yaml color term file.

        Returns
        -------
        yaml_file :  `str`
            Name of yaml file with color term.
        """
        gaia_ref_info = self.gaia_reference_class()
        cat_info_list = [cat_info() for cat_info in self.catalog_info_class_list]
        uband_ref_info = self.uband_ref_class()
        uband_slr_info = self.uband_slr_class()

        ra_min, ra_max, dec_min, dec_max = self.ra_dec_range

        box = sphgeom.Box.fromDegrees(ra_min, dec_min, ra_max, dec_max)

        pixelization = sphgeom.HtmPixelization(7)
        rs = pixelization.envelope(box)

        indices = []
        for (begin, end) in rs:
            indices.extend(range(begin, end))

        gaia_stars_all = read_uband_combined_catalog(
            gaia_ref_info,
            cat_info_list,
            uband_ref_info,
            indices,
            testing_mode=self.testing_mode,
        )

        _combined_info = _CombinedInfo()

        slr_band = _combined_info.get_slr_band("u")

        mag_color = _combined_info.get_mag_colors(gaia_stars_all, "u")
        flux_target = gaia_stars_all[_combined_info.get_flux_field("u")]
        flux_cat = gaia_stars_all[_combined_info.get_flux_field(slr_band)]

        color_range = uband_slr_info.get_color_range("u")

        nodes = np.linspace(color_range[0], color_range[1], self.n_nodes)

        # All selections have already been applied, we only
        # need to check that the stars have u/g/r fluxes.
        selected = np.isfinite(gaia_stars_all["ref_u_flux"])
        for band in _combined_info.get_color_bands("u"):
            selected &= (np.isfinite(gaia_stars_all[f"{band}_flux"]))

        # Do the first round of the SLR fit.
        fitter = ColortermSplineFitter(
            mag_color[selected],
            flux_target[selected],
            flux_cat[selected],
            nodes,
            fit_flux_offset=self.do_fit_flux_offset,
        )

        p0 = fitter.estimate_p0()
        pars = fitter.fit(p0)

        band_1, band_2 = uband_slr_info.get_color_bands("u")

        if self.do_fit_flux_offset:
            spline_values = pars[:-1]
            flux_offset = pars[-1]
        else:
            spline_values = pars
            flux_offset = 0.0

        # The slr names will be bands.
        colorterm_init = ColortermSpline(
            uband_slr_info.name,
            uband_ref_info.name,
            band_1,
            band_2,
            slr_band,
            nodes,
            spline_values,
            flux_offset=flux_offset,
        )

        mag_nodes = None
        mag_spline_values = None

        if self.do_fit_mag_offsets:
            model_flux = colorterm_init.apply(
                gaia_stars_all[_combined_info.get_flux_field(band_1)],
                gaia_stars_all[_combined_info.get_flux_field(band_2)],
                gaia_stars_all[_combined_info.get_flux_field(slr_band)],
            )

            selected2 = selected & np.isfinite(model_flux)
            model_flux[~selected2] = np.nan

            # The ``mag_offset_model_flux`` is the flux that we want to target.
            mag_offset_model_flux = flux_target.copy()

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model_mag = (model_flux*units.nJy).to_value(units.ABmag)

            # Magnitude nodes will cover the range of training.
            mag_node_range = [np.min(model_mag[selected2]),
                              np.max(model_mag[selected2])]

            mag_nodes = np.linspace(mag_node_range[0], mag_node_range[1], self.n_mag_nodes)

            mag_fitter = MagSplineFitter(
                np.array(mag_offset_model_flux[selected2]),
                np.array(model_flux[selected2]),
                mag_nodes,
            )
            mag_p0 = mag_fitter.estimate_p0()
            mag_spline_values = mag_fitter.fit(mag_p0)

        colorterm = ColortermSpline(
            uband_slr_info.name,
            uband_ref_info.name,
            band_1,
            band_2,
            slr_band,
            nodes,
            spline_values,
            flux_offset=flux_offset,
            mag_nodes=mag_nodes,
            mag_spline_values=mag_spline_values,
        )

        yaml_file = f"{uband_slr_info.name}_to_{uband_ref_info.name}_band_u.yaml"
        colorterm.save(yaml_file, overwrite=overwrite)

        yaml_files = [yaml_file]

        if do_plots:
            ratio_extent = np.nanpercentile(flux_cat[selected]/flux_target[selected], [0.5, 99.5])

            xlabel = f"{band_1} - {band_2}"
            ylabel = f"{slr_band}/{uband_ref_info.name}_u"

            xvals = np.linspace(color_range[0], color_range[1], 1000)
            yvals = 1./np.array(colorterm.spline.interpolate(xvals))

            plt.clf()
            plt.hexbin(
                mag_color[selected],
                (flux_cat[selected] - flux_offset)/flux_target[selected],
                bins="log",
                extent=[color_range[0], color_range[1], ratio_extent[0], ratio_extent[1]],
            )
            plt.plot(xvals, yvals, "r-")
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title("u-band SLR color term")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                plt.tight_layout()
            plt.savefig(f"transformed_to_{uband_ref_info.name}_band_g_slr.png")

            flux_target_corr = colorterm.apply(
                gaia_stars_all[_combined_info.get_flux_field(band_1)],
                gaia_stars_all[_combined_info.get_flux_field(band_2)],
                gaia_stars_all[_combined_info.get_flux_field(slr_band)],
            )
            resid = (flux_target_corr[selected] - flux_target[selected])/flux_target[selected]

            resid_extent = np.nanpercentile(resid, [0.5, 99.5])

            xlabel2 = f"mag_u ({uband_ref_info.name})"

            mag_target_selected = (np.array(flux_target[selected])*units.nJy).to_value(units.ABmag)
            mag_extent = np.nanpercentile(mag_target_selected, [0.5, 99.5])

            plt.clf()
            plt.hexbin(
                mag_target_selected,
                resid,
                bins='log',
                extent=[mag_extent[0], mag_extent[1], resid_extent[0], resid_extent[1]],
            )
            plt.plot(mag_extent, [0, 0], 'r:')
            plt.xlabel(xlabel2)
            plt.ylabel(ylabel + " - model")
            plt.title("uSLR flux residuals")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                plt.tight_layout()
            plt.savefig(f"transformed_to_{uband_ref_info.name}_band_g_slr_flux_residuals.png")

            # Additional plots for magnitude offset pars.
            if self.do_fit_mag_offsets:
                xvals = np.linspace(mag_nodes[0], mag_nodes[-1], 1000)
                yvals = 1./np.array(colorterm.mag_spline.interpolate(xvals))

                ratio_extent = np.nanpercentile(model_flux[selected2]/mag_offset_model_flux[selected2],
                                                [0.5, 99.5])

                plt.clf()
                plt.hexbin(
                    model_mag[selected2],
                    model_flux[selected2]/mag_offset_model_flux[selected2],
                    bins="log",
                    extent=[mag_nodes[0], mag_nodes[-1], ratio_extent[0], ratio_extent[1]],
                )
                plt.plot(xvals, yvals, "r-")
                plt.xlabel("u")
                plt.ylabel(f"uSLR/{uband_ref_info.name}_u")
                plt.title("uSLR Magnitude Residual Term")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    plt.tight_layout()
                plt.savefig(f"transformed_to_{uband_ref_info.name}_band_g_slr_mag_offset.png")

        return yaml_files
