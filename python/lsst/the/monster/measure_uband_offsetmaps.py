import os
import numpy as np
from astropy import units
import hpgeom as hpg
import healsparse as hsp
import skyproj
from smatch import Matcher
import esutil
import scipy.optimize
import matplotlib.pyplot as plt

import lsst.sphgeom as sphgeom

from .refcats import GaiaDR3Info, GaiaXPuInfo, DESInfo, SkyMapperInfo, PS1Info, VSTInfo
from .splinecolorterms import ColortermSpline
from .utils import read_stars


__all__ = [
    "UbandOffsetMapMaker",
]


class UbandOffsetMapMaker:
    def __init__(
        self,
        gaia_reference_class=GaiaDR3Info,
        catalog_info_class_list=[VSTInfo, SkyMapperInfo,
                                 PS1Info, DESInfo],
        uband_ref_class=GaiaXPuInfo,
        uband_slr_class=DESInfo,
        testing_mode=False,
        nside=32,
        nside_coarse=8,
        htm_level=7,
    ):
        self.gaia_reference_class = gaia_reference_class
        self.catalog_info_class_list = catalog_info_class_list
        self.uband_ref_class = uband_ref_class
        self.uband_slr_class = uband_slr_class
        self.testing_mode = testing_mode
        self.nside = nside
        self.nside_coarse = nside_coarse
        self.htm_level = 7

    def measure_uband_offset_map(self, overwrite=False):
        """Measure the u-band offset map, and save to a healsparse file.

        Parameters
        ----------
        overwrite : `bool`, optional
            Overwrite an existing map file.

        Returns
        -------
        hsp_file : `str`
            Name of healsparse file with maps.
        """
        gaia_ref_info = self.gaia_reference_class()
        cat_info_list = [cat_info() for cat_info in self.catalog_info_class_list]
        uband_ref_info = self.uband_ref_class()
        uband_slr_info = self.uband_slr_class()

        # These are the bands we are using to transform to u.
        u_slr_bands = ["g", "r"]

        fname = f"uband_offset_map_{uband_ref_info.name}.hsp"

        if os.path.isfile(fname):
            if overwrite:
                print(f"Found existing {fname}; will overwrite.")
            else:
                print(f"Found existing {fname}; overwrite=False so no need to remake map.")
                return fname

        print("Computing u-band offset map.")

        # Read in the colorterm for the reference uband.
        ref_colorterm_filename = uband_ref_info.colorterm_file("u")
        ref_colorterm_spline = ColortermSpline.load(ref_colorterm_filename)

        # And the SLR colorterm.
        slr_colorterm_filename = os.path.join(
            uband_slr_info._colorterm_path,
            f"{uband_slr_info.name}_to_GaiaXP_band_u.yaml",
        )
        slr_colorterm_spline = ColortermSpline.load(slr_colorterm_filename)

        healpix_pixelization = sphgeom.HealpixPixelization(hpg.nside_to_order(self.nside_coarse))
        htm_pixelization = sphgeom.HtmPixelization(self.htm_level)

        if not self.testing_mode:
            pixels = np.arange(hpg.nside_to_npixel(self.nside_coarse), dtype=np.int64)
        else:
            box = sphgeom.Box.fromDegrees(20, -40, 40, -20)
            rs = healpix_pixelization.envelope(box)
            pixels = []
            for (begin, end) in rs:
                pixels.extend(range(begin, end))

        dtype = [("nref_u", "i4"),
                 ("nslr_u", "i4"),
                 ("nmatch_u", "i4"),
                 ("offset_u", "f4")]

        offset_map = hsp.HealSparseMap.make_empty(32, self.nside, dtype, primary="nmatch_u")

        for pixel in pixels:
            healpix_poly = healpix_pixelization.pixel(pixel)

            htm_pixel_range = htm_pixelization.envelope(healpix_poly)
            htm_pixel_list = []
            for r in htm_pixel_range.ranges():
                htm_pixel_list.extend(range(r[0], r[1]))

            # Read in the overall reference catalog.
            gaia_stars_all = read_stars(gaia_ref_info.path, htm_pixel_list, allow_missing=self.testing_mode)
            if len(gaia_stars_all) == 0:
                continue

            nan_column = np.full(len(gaia_stars_all["id"]), np.nan)
            # We need to add in columns for flux/err for u, g, r.
            for band in u_slr_bands:
                gaia_stars_all.add_column(nan_column, name=f"{band}_flux")
                gaia_stars_all.add_column(nan_column, name=f"{band}_fluxErr")

            gaia_stars_all.add_column(nan_column, name="ref_u_flux")
            gaia_stars_all.add_column(nan_column, name="ref_u_fluxErr")
            gaia_stars_all.add_column(nan_column, name="slr_u_flux")
            gaia_stars_all.add_column(nan_column, name="slr_u_fluxErr")

            # Loop over the refcats.
            for cat_info in cat_info_list:
                # We read from the "write_path" which has the (DES)
                # transformed catalog.
                cat_stars = read_stars(cat_info.write_path, htm_pixel_list, allow_missing=True)
                if len(cat_stars) == 0:
                    continue

                if self.testing_mode:
                    # We need to hack the catalogs for the test names (sorry).
                    for name in cat_stars.dtype.names:
                        if (substr := "_" + cat_info.ORIG_NAME_FOR_TEST + "_") in name:
                            new_name = name.replace(substr, "_" + cat_info.NAME + "_")
                            cat_stars.rename_column(name, new_name)

                for band in u_slr_bands:
                    cat_flux_field = cat_info.get_transformed_flux_field(band)
                    selected = np.isfinite(cat_stars[cat_flux_field])

                    if len(selected) == 0:
                        continue

                    cat_stars_selected = cat_stars[selected]

                    a, b = esutil.numpy_util.match(gaia_stars_all["id"],
                                                   cat_stars_selected["GaiaDR3_id"])
                    gaia_stars_all[f"{band}_flux"][a] = cat_stars_selected[cat_flux_field][b]
                    gaia_stars_all[f"{band}_fluxErr"][a] = cat_stars_selected[cat_flux_field + "Err"][b]

            # And read in the reference catalog, transform, and fill in.
            uband_ref_stars = read_stars(uband_ref_info.path, htm_pixel_list, allow_missing=True)
            if len(uband_ref_stars) == 0:
                # This would be surprising.
                print(f"No uband ref stars found for coarse pixel {pixel}!")
                continue

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

            # We need stars that have good g-r colors for the SLR
            # measurements.
            slr_selected = np.isfinite(gaia_stars_all["g_flux"]) & np.isfinite(gaia_stars_all["r_flux"])
            gaia_stars_all = gaia_stars_all[slr_selected]

            # Apply the SLR to compute slr_u_flux, slr_u_fluxErr
            gaia_stars_all["slr_u_flux"] = slr_colorterm_spline.apply(
                gaia_stars_all["g_flux"],
                gaia_stars_all["r_flux"],
                gaia_stars_all["g_flux"],
            )
            gaia_stars_all["slr_u_fluxErr"] = gaia_stars_all["slr_u_flux"] * (
                gaia_stars_all["g_fluxErr"] / gaia_stars_all["g_flux"]
            )

            # We only care about stars that have valid
            # slr u flux OR ref u flux.
            u_selected = np.isfinite(gaia_stars_all["ref_u_flux"]) | np.isfinite(gaia_stars_all["slr_u_flux"])
            gaia_stars_all = gaia_stars_all[u_selected]

            ipnest = hpg.angle_to_pixel(self.nside, gaia_stars_all["coord_ra"], gaia_stars_all["coord_dec"])
            h, rev = esutil.stat.histogram(ipnest, rev=True)

            pixuse, = np.where(h > 0)

            for pixind in pixuse:
                i1a = rev[rev[pixind]: rev[pixind + 1]]

                element = np.zeros(1, dtype=dtype)

                # Number of reference u-band stars?
                selected_uref = np.isfinite(gaia_stars_all["ref_u_flux"][i1a])
                element["nref_u"] = selected_uref.sum()

                # Number of SLR u-band stars?
                selected_uslr = np.isfinite(gaia_stars_all["slr_u_flux"][i1a])
                element["nslr_u"] = selected_uslr.sum()

                # Number of matched stars?
                selected_matched = selected_uref & selected_uslr
                element["nmatch_u"] = selected_matched.sum()

                if element["nmatch_u"] > 1:
                    slr_u_mag = (gaia_stars_all["slr_u_flux"][i1a[selected_matched]]
                                 * units.nJy).to_value(units.ABmag)
                    ref_u_mag = (gaia_stars_all["ref_u_flux"][i1a[selected_matched]]
                                 * units.nJy).to_value(units.ABmag)
                    element["offset_u"] = np.median(slr_u_mag - ref_u_mag)

                offset_map[ipnest[i1a[0]]] = element

        offset_map.write(fname, clobber=overwrite)

        return fname

    def plot_uband_offset_maps(self, hsp_file):
        """Plot uband offset maps (and histograms) from a map file.

        Parameters
        ----------
        hsp_file : `str`
            Name of healsparse file with maps.
        """
        offset_map = hsp.HealSparseMap.read(hsp_file)

        def gauss(x, *p):
            A, mu, sigma = p
            return A*np.exp(-(x-mu)**2./(2.*sigma**2.))

        nmatch = offset_map["nmatch_u"].copy()
        offset = offset_map["offset_u"].copy()

        valid_pixels = nmatch.valid_pixels
        bad, = np.where(nmatch[valid_pixels] < 3)
        offset[valid_pixels[bad]] = None

        valid_pixels, valid_ra, valid_dec = offset.valid_pixels_pos(return_pixels=True)

        # First we plot the full u-band offset map.
        plt.clf()
        fig = plt.figure(figsize=(16, 6))
        ax = fig.add_subplot(111)

        sp = skyproj.McBrydeSkyproj(ax=ax)
        sp.draw_hspmap(offset*1000., zoom=True)
        sp.draw_colorbar(label="SLR u - XP u (mmag)")
        fig.savefig("uslr-uxp_full_map.png")
        plt.close(fig)

        # And cut low Galactic latitude regions.
        l, b = esutil.coords.eq2gal(valid_ra, valid_dec)
        low, = np.where(np.abs(b) < 30.0)

        offset[valid_pixels[low]] = None

        # And plot a high galactic latitude map.
        plt.clf()
        fig = plt.figure(figsize=(16, 6))
        ax = fig.add_subplot(111)

        sp = skyproj.McBrydeSkyproj(ax=ax)
        sp.draw_hspmap(offset*1000., zoom=True)
        sp.draw_colorbar(label="SLR u - XP u (mmag)")
        fig.savefig("uslr-uxp_highglat_map.png")
        plt.close(fig)

        # Fit a Gaussian to the high galactic latitude.
        data = offset[offset.valid_pixels]*1000.
        if (len(data) < 10):
            # No point in trying to make a histogram.
            return

        plt.clf()
        vmin, vmax = np.percentile(data, q=[1.0, 99.0])
        nbins = 100
        n, b, p = plt.hist(
            data,
            bins=np.linspace(vmin, vmax, nbins),
            histtype="step",
            color="blue",
            lw=1.5,
        )

        p0 = [data.size, np.mean(data), np.std(data)]
        hist_fit_x = (np.array(b[0: -1]) + np.array(b[1:]))/2.
        hist_fit_y = np.array(n)
        try:
            coeff, var_mat = scipy.optimize.curve_fit(gauss, hist_fit_x, hist_fit_y, p0=p0)
        except RuntimeError:
            return

        xvals = np.linspace(vmin, vmax, 1000)
        yvals = gauss(xvals, *coeff)

        plt.plot(xvals, yvals, "k--", linewidth=3)
        plt.xlabel("SLR u - XP u (mmag)")
        plt.ylabel(f"Number of nside={offset.nside_sparse} pixels")
        plt.annotate(
            r"$\mu = %.1f\,\mathrm{mmag}$" % (coeff[1]),
            (0.99, 0.99),
            fontsize=14,
            xycoords="axes fraction",
            ha="right",
            va="top",
        )
        plt.annotate(
            r"$\sigma = %.1f\,\mathrm{mmag}$" % (coeff[2]),
            (0.99, 0.92),
            fontsize=14,
            xycoords="axes fraction",
            ha="right",
            va="top",
        )

        plt.savefig("uslr-uxp_highglat_hist.png")
