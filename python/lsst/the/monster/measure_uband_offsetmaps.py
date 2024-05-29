import os
import numpy as np
from astropy import units
import hpgeom as hpg
import healsparse as hsp
import skyproj
import esutil
import scipy.optimize
import matplotlib.pyplot as plt
from smatch import Matcher

import lsst.sphgeom as sphgeom

from .refcats import GaiaDR3Info, GaiaXPuInfo, DESInfo, SkyMapperInfo, PS1Info, VSTInfo
from .splinecolorterms import ColortermSpline
from .measure_uband_slr_colorterm import read_uband_combined_catalog
from .utils import read_stars


__all__ = [
    "UBandOffsetMapMaker",
]


class UBandOffsetMapMaker:
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
        """Measure the SLR computed u-band offset map, and save to a
        healsparse file.

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

        fname = f"uband_offset_map_{uband_ref_info.name}.hsp"

        if os.path.isfile(fname):
            if overwrite:
                print(f"Found existing {fname}; will overwrite.")
            else:
                print(f"Found existing {fname}; overwrite=False so no need to remake map.")
                return fname

        print("Computing u-band offset map.")

        # Read in the SLR colorterm.
        slr_colorterm_filename = os.path.join(
            uband_slr_info._colorterm_path,
            f"{uband_slr_info.name}_to_GaiaXP_band_u.yaml",
        )
        print("Using SLR colorterm file: ", slr_colorterm_filename)
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
            print("Working on pixel ", pixel)
            healpix_poly = healpix_pixelization.pixel(pixel)

            htm_pixel_range = htm_pixelization.envelope(healpix_poly)
            htm_pixel_list = []
            for r in htm_pixel_range.ranges():
                htm_pixel_list.extend(range(r[0], r[1]))

            # Read in the combined catalog.
            gaia_stars_all = read_uband_combined_catalog(
                gaia_ref_info,
                cat_info_list,
                uband_ref_info,
                htm_pixel_list,
                testing_mode=self.testing_mode,
            )
            if len(gaia_stars_all) == 0:
                continue

            # Cut down to those that are in the coarse pixel.
            use, = np.where(hpg.angle_to_pixel(
                self.nside_coarse,
                gaia_stars_all["coord_ra"],
                gaia_stars_all["coord_dec"]) == pixel)
            if use.size == 0:
                continue
            gaia_stars_all = gaia_stars_all[use]

            nan_column = np.full(len(gaia_stars_all), np.nan)
            # We need to add a column for the slr u flux and error.
            gaia_stars_all.add_column(nan_column, name="slr_u_flux")
            gaia_stars_all.add_column(nan_column, name="slr_u_fluxErr")

            band_1 = slr_colorterm_spline.source_color_field_1
            band_2 = slr_colorterm_spline.source_color_field_2
            source_band = slr_colorterm_spline.source_field

            # At the minimum we will only use stars that we can do the SLR
            # estimation of u flux.
            slr_selected = (np.isfinite(gaia_stars_all[f"{band_1}_flux"])
                            & np.isfinite(gaia_stars_all[f"{band_2}_flux"])
                            & np.isfinite(gaia_stars_all[f"{source_band}_flux"]))
            gaia_stars_all = gaia_stars_all[slr_selected]

            slr_orig_flux = gaia_stars_all[f"{source_band}_flux"]
            slr_orig_flux_err = gaia_stars_all[f"{source_band}_fluxErr"]

            gaia_stars_all["slr_u_flux"] = slr_colorterm_spline.apply(
                gaia_stars_all[f"{band_1}_flux"],
                gaia_stars_all[f"{band_2}_flux"],
                slr_orig_flux,
            )
            gaia_stars_all["slr_u_fluxErr"] = gaia_stars_all["slr_u_flux"] * (
                slr_orig_flux_err / slr_orig_flux
            )

            # Now we select stars that have valid slr u OR ref u.
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

    def measure_uband_offset_map_direct(self, overwrite=False):
        """Measure the direct u-band offset map, and save to a
        healsparse file.

        Parameters
        ----------
        overwrite : `bool`, optional
            Overwrite an existing map file.

        Returns
        -------
        hsp_file : `str`
            Name of healsparse file with maps.
        """
        if len(self.catalog_info_class_list) != 1:
            raise RuntimeError("Direct computation can only use 1 catalog info class.")

        gaia_ref_info = self.gaia_reference_class()
        cat_info = self.catalog_info_class_list[0]()
        uband_ref_info = self.uband_ref_class()

        fname = f"uband_offset_map_direct_{cat_info.name}-{uband_ref_info.name}.hsp"

        if os.path.isfile(fname):
            if overwrite:
                print(f"Found existing {fname}; will overwrite.")
            else:
                print(f"Found existing {fname}; overwrite=False so no need to remake map.")
                return fname

        print("Computing direct u-band offset map.")

        healpix_pixelization = sphgeom.HealpixPixelization(hpg.nside_to_order(self.nside_coarse))
        htm_pixelization = sphgeom.HtmPixelization(self.htm_level)

        if not self.testing_mode:
            pixels = np.arange(hpg.nside_to_npixel(self.nside_coarse), dtype=np.int64)
        else:
            box = sphgeom.Box.fromDegrees(150, 10, 180, 30)
            rs = healpix_pixelization.envelope(box)
            pixels = []
            for (begin, end) in rs:
                pixels.extend(range(begin, end))

        dtype = [("nref_u", "i4"),
                 ("ncat_u", "i4"),
                 ("nmatch_u", "i4"),
                 ("offset_u", "f4")]

        offset_map = hsp.HealSparseMap.make_empty(32, self.nside, dtype, primary="nmatch_u")

        for pixel in pixels:
            print("Working on pixel ", pixel)
            healpix_poly = healpix_pixelization.pixel(pixel)

            htm_pixel_range = htm_pixelization.envelope(healpix_poly)
            htm_pixel_list = []
            for r in htm_pixel_range.ranges():
                htm_pixel_list.extend(range(r[0], r[1]))

            # Read in the reference catalog. We are using the code in a
            # mode to read in the reference u-band and associate with
            # Gaia only. Association with our comparison catalog is
            # below.
            gaia_stars_all = read_uband_combined_catalog(
                gaia_ref_info,
                [],
                uband_ref_info,
                htm_pixel_list,
                testing_mode=self.testing_mode,
            )
            if len(gaia_stars_all) == 0:
                continue

            # Read in the comparison catalog.
            uband_stars = read_stars(cat_info.path, htm_pixel_list, allow_missing=True)

            # Apply the transformation.
            colorterm_spline = ColortermSpline.load(cat_info.colorterm_file("u"))

            band_1, band_2 = cat_info.get_color_bands("u")
            uband_orig_flux = uband_stars[cat_info.get_flux_field("u")]
            uband_orig_flux_err = uband_stars[cat_info.get_flux_field("u") + "Err"]
            uband_model_flux = colorterm_spline.apply(
                uband_stars[cat_info.get_flux_field(band_1)],
                uband_stars[cat_info.get_flux_field(band_2)],
                uband_orig_flux,
            )

            uband_selected = cat_info.select_stars(uband_stars, "u")
            uband_selected &= np.isfinite(uband_model_flux)

            uband_stars = uband_stars[uband_selected]
            uband_orig_flux = uband_orig_flux[uband_selected]
            uband_orig_flux_err = uband_orig_flux_err[uband_selected]
            uband_model_flux = uband_model_flux[uband_selected]

            with Matcher(gaia_stars_all["coord_ra"], gaia_stars_all["coord_dec"]) as m:
                idx, i1, i2, d = m.query_knn(
                    uband_stars["coord_ra"],
                    uband_stars["coord_dec"],
                    distance_upper_bound=0.5/3600.,
                    return_indices=True,
                )

            nan_column = np.full(len(gaia_stars_all), np.nan)
            gaia_stars_all.add_column(nan_column, name="cat_u_flux")
            gaia_stars_all.add_column(nan_column, name="cat_u_fluxErr")

            gaia_stars_all["cat_u_flux"][i1] = uband_model_flux[i2]
            gaia_stars_all["cat_u_fluxErr"][i1] = uband_model_flux[i2] * (
                uband_orig_flux_err[i2]/uband_orig_flux[i2]
            )

            # Now we select stars that have valid u band in both
            u_selected = np.isfinite(gaia_stars_all["ref_u_flux"]) & np.isfinite(gaia_stars_all["cat_u_flux"])
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
                selected_uslr = np.isfinite(gaia_stars_all["cat_u_flux"][i1a])
                element["ncat_u"] = selected_uslr.sum()

                # Number of matched stars?
                selected_matched = selected_uref & selected_uslr
                element["nmatch_u"] = selected_matched.sum()

                if element["nmatch_u"] > 1:
                    slr_u_mag = (gaia_stars_all["cat_u_flux"][i1a[selected_matched]]
                                 * units.nJy).to_value(units.ABmag)
                    ref_u_mag = (gaia_stars_all["ref_u_flux"][i1a[selected_matched]]
                                 * units.nJy).to_value(units.ABmag)
                    element["offset_u"] = np.median(slr_u_mag - ref_u_mag)

                offset_map[ipnest[i1a[0]]] = element

        offset_map.write(fname, clobber=overwrite)

        return fname

    def plot_uband_offset_maps(self, hsp_file, mode="slr"):
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

        if mode == "slr":
            label = "SLR u - XP u (mmag)"
            fname_base = "uslr-uxp"
        else:
            label = "XP u - SDSS u (mmag)"
            fname_base = "uxp-usdss"

        # First we plot the full u-band offset map.
        plt.clf()
        fig = plt.figure(figsize=(16, 6))
        ax = fig.add_subplot(111)

        sp = skyproj.McBrydeSkyproj(ax=ax)
        sp.draw_hspmap(offset*1000., zoom=True)
        sp.draw_colorbar(label=label)
        fig.savefig(f"{fname_base}_full_map.png")
        plt.close(fig)

        if mode == "slr":
            # Plot the number of SLR stars in the map.
            plt.clf()
            fig = plt.figure(figsize=(16, 6))
            ax = fig.add_subplot(111)

            sp = skyproj.McBrydeSkyproj(ax=ax)
            sp.draw_hspmap(offset_map["nslr_u"], zoom=True)
            sp.draw_colorbar(label=f"# SLR Stars (nside {self.nside})")
            fig.savefig("uslr_nstar.png")
            plt.close(fig)
        else:
            # Plot the number of matched stars.
            plt.clf()
            fig = plt.figure(figsize=(16, 6))
            ax = fig.add_subplot(111)

            sp = skyproj.McBrydeSkyproj(ax=ax)
            sp.draw_hspmap(offset_map["nmatch_u"], zoom=True)
            sp.draw_colorbar(label=f"# Matched Stars (nside {self.nside})")
            fig.savefig("umatch_xp_sdss_nstar.png")
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
        sp.draw_colorbar(label=label)
        fig.savefig(f"{fname_base}_highglat_map.png")
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
        plt.xlabel(label)
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

        plt.savefig(f"{fname_base}_highglat_hist.png")
