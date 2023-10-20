import numpy as np
from astropy import units
import esutil
import hpgeom as hpg
import healsparse as hsp
import skyproj
import scipy.optimize
import matplotlib.pyplot as plt

import lsst.sphgeom as sphgeom

from .utils import read_stars
from .refcats import GaiaXPInfo, DESInfo, SkyMapperInfo, PS1Info, VSTInfo


__all__ = [
    "OffsetMapMaker",
    "GaiaXPMinusDESOffsetMapMaker",
    "PS1MinusDESOffsetMapMaker",
    "SkyMapperMinusDESOffsetMapMaker",
    "VSTMinusDESOffsetMapMaker",
    "PS1MinusGaiaXPOffsetMapMaker",
    "SkyMapperMinusGaiaXPOffsetMapMaker",
    "SkyMapperMinusPS1OffsetMapMaker",
    "VSTMinusGaiaXPOffsetMapMaker",
]


class OffsetMapMaker:
    MinuendInfoClass = None
    SubtrahendInfoClass = None

    # Does this have low Galactic latitude area?
    has_low_glat = False

    # Name of ID to use for matching.
    match_id_name = "GaiaDR3_id"

    @property
    def nside(self):
        return 128

    @property
    def nside_coarse(self):
        return 8

    @property
    def htm_level(self):
        return 7

    def measure_offset_map(self, bands=None, overwrite=False):
        """Measure the offset map, and save to a healsparse file.

        Parameters
        ----------
        bands : `list` [`str`]
            Name of bands to compute. If not specified, will use
            the overlap in the two catalogs.
        overwrite : `bool`, optional
            Overwrite an existing map file.

        Returns
        -------
        hsp_file : `str`
            Name of healsparse file with maps.
        """
        minuend_info = self.MinuendInfoClass()
        subtrahend_info = self.SubtrahendInfoClass()

        minuend_path = minuend_info.write_path
        subtrahend_path = subtrahend_info.write_path

        if bands is None:
            minuend_bands = set(minuend_info.bands)
            subtrahend_bands = set(subtrahend_info.bands)
            bands = sorted(list(minuend_bands.intersection(subtrahend_bands)))

        print("Using bands: ", bands)

        southern_pixels = hpg.query_box(self.nside_coarse, 0.0, 360.0, -90.0, 35.0)

        healpix_pixelization = sphgeom.HealpixPixelization(hpg.nside_to_order(self.nside_coarse))
        htm_pixelization = sphgeom.HtmPixelization(self.htm_level)

        dtype = [("nmatch", "i4"),]
        for band in bands:
            dtype.extend([
                (f"offset_{band}", "f4"),
                (f"ngood_{band}", "i4"),
            ])

        offset_map = hsp.HealSparseMap.make_empty(32, self.nside, dtype, primary="nmatch")

        for southern_pixel in southern_pixels:
            healpix_poly = healpix_pixelization.pixel(southern_pixel)

            htm_pixel_range = htm_pixelization.envelope(healpix_poly)
            htm_pixel_list = []
            for r in htm_pixel_range.ranges():
                htm_pixel_list.extend(range(r[0], r[1]))

            minuend_stars = read_stars(minuend_path, htm_pixel_list, allow_missing=True)
            if len(minuend_stars) == 0:
                continue
            subtrahend_stars = read_stars(subtrahend_path, htm_pixel_list, allow_missing=True)
            if len(subtrahend_stars) == 0:
                continue

            print("Working on ", southern_pixel)
            a, b = esutil.numpy_util.match(
                minuend_stars[self.match_id_name],
                subtrahend_stars[self.match_id_name],
            )

            if len(a) == 0:
                continue

            minuend_stars = minuend_stars[a]
            subtrahend_stars = subtrahend_stars[b]

            # Cut down to those in the healpix pixel
            use, = np.where(hpg.angle_to_pixel(
                self.nside_coarse,
                subtrahend_stars["coord_ra"],
                subtrahend_stars["coord_dec"]) == southern_pixel)
            if use.size == 0:
                continue
            minuend_stars = minuend_stars[use]
            subtrahend_stars = subtrahend_stars[use]

            ipnest = hpg.angle_to_pixel(
                self.nside,
                subtrahend_stars["coord_ra"],
                subtrahend_stars["coord_dec"],
            )

            h, rev = esutil.stat.histogram(ipnest, rev=True)

            pixuse, = np.where(h > 0)

            for pixind in pixuse:
                i1a = rev[rev[pixind]: rev[pixind + 1]]

                element = np.zeros(1, dtype=dtype)
                element["nmatch"] = len(i1a)

                for band in bands:
                    minuend_flux = minuend_stars[f"decam_{band}_from_{minuend_info.name}_flux"][i1a]
                    minuend_flux_err = minuend_stars[f"decam_{band}_from_{minuend_info.name}_fluxErr"][i1a]
                    subtrahend_flux = subtrahend_stars[f"decam_{band}_from_{subtrahend_info.name}_flux"][i1a]
                    subtrahend_flux_err = subtrahend_stars[
                        f"decam_{band}_from_{subtrahend_info.name}_fluxErr"
                    ][i1a]

                    gd = (np.isfinite(minuend_flux)
                          & np.isfinite(minuend_flux_err)
                          & np.isfinite(subtrahend_flux)
                          & np.isfinite(subtrahend_flux_err))

                    element[f"ngood_{band}"] = gd.sum()

                    if gd.sum() == 0:
                        continue

                    element[f"offset_{band}"] = np.median(
                        minuend_flux[gd].quantity.to_value(units.ABmag)
                        - subtrahend_flux[gd].quantity.to_value(units.ABmag)
                    )

                offset_map[ipnest[i1a[0]]] = element

        fname = f"offset_map_{minuend_info.name}-{subtrahend_info.name}.hsp"
        offset_map.write(fname)

        return fname

    def plot_offset_maps(self, hsp_file):
        """Plot offset maps (and histograms) from a map file.

        Parameters
        ----------
        hsp_file : `str`
            Name of healsparse file with maps.
        """
        minuend_info = self.MinuendInfoClass()
        subtrahend_info = self.SubtrahendInfoClass()

        offset_map = hsp.HealSparseMap.read(hsp_file)

        bands = []
        for name in offset_map.dtype.names:
            if name.startswith("ngood"):
                parts = name.split("_")
                bands.append(parts[1])

        def gauss(x, *p):
            A, mu, sigma = p
            return A*np.exp(-(x-mu)**2./(2.*sigma**2.))

        for band in bands:
            ngood_band = offset_map[f"ngood_{band}"].copy()
            offset_band = offset_map[f"offset_{band}"].copy()

            valid_pixels = ngood_band.valid_pixels
            bad, = np.where(ngood_band[valid_pixels] < 3)
            offset_band[valid_pixels[bad]] = None

            valid_pixels, valid_ra, valid_dec = offset_band.valid_pixels_pos(return_pixels=True)

            if self.has_low_glat:
                # Make a full sky map

                plt.clf()
                fig = plt.figure(figsize=(18, 6))
                ax = fig.add_subplot(111)

                sp = skyproj.McBrydeSkyproj(ax=ax)
                sp.draw_hspmap(offset_band*1000., zoom=True)
                sp.draw_colorbar(label=f"{minuend_info.name} - {subtrahend_info.name} {band} (mmag)")
                fig.savefig(f"{minuend_info.name}-{subtrahend_info.name}_fullmap_{band}.png")
                plt.close(fig)

                # Now cut out low Galactic latitude regions.
                l, b = esutil.coords.eq2gal(valid_ra, valid_dec)
                low, = np.where(np.abs(b) < 30.0)

                offset_band[valid_pixels[low]] = None

            plt.clf()
            fig = plt.figure(figsize=(18, 6))
            ax = fig.add_subplot(111)

            sp = skyproj.McBrydeSkyproj(ax=ax)
            sp.draw_hspmap(offset_band*1000., zoom=True)
            sp.draw_colorbar(label=f"{minuend_info.name} - {subtrahend_info.name} {band} (mmag)")
            fig.savefig(f"{minuend_info.name}-{subtrahend_info.name}_highglat_{band}.png")
            plt.close(fig)

            # Do a histogram and fit a Gaussian to it.
            plt.clf()
            data = offset_band[offset_band.valid_pixels]*1000.
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
                # During testing, this may not get a fit.
                continue

            xvals = np.linspace(vmin, vmax, 1000)
            yvals = gauss(xvals, *coeff)

            plt.plot(xvals, yvals, "k--", linewidth=3)
            plt.xlabel(f"{minuend_info.name} - {subtrahend_info.name} {band} (mmag)")
            plt.ylabel(f"Number of nside={offset_band.nside_sparse} pixels")
            plt.annotate(
                r"$\sigma = %.1f\,\mathrm{mmag}$" % (coeff[2]),
                (0.99, 0.99),
                fontsize=14,
                xycoords="axes fraction",
                ha="right",
                va="top",
            )
            plt.savefig(f"{minuend_info.name}-{subtrahend_info.name}_highglat_hist_{band}.png")


class GaiaXPMinusDESOffsetMapMaker(OffsetMapMaker):
    MinuendInfoClass = GaiaXPInfo
    SubtrahendInfoClass = DESInfo


class PS1MinusDESOffsetMapMaker(OffsetMapMaker):
    MinuendInfoClass = PS1Info
    SubtrahendInfoClass = DESInfo


class SkyMapperMinusDESOffsetMapMaker(OffsetMapMaker):
    MinuendInfoClass = SkyMapperInfo
    SubtrahendInfoClass = DESInfo


class VSTMinusDESOffsetMapMaker(OffsetMapMaker):
    MinuendInfoClass = VSTInfo
    SubtrahendInfoClass = DESInfo


class PS1MinusGaiaXPOffsetMapMaker(OffsetMapMaker):
    MinuendInfoClass = PS1Info
    SubtrahendInfoClass = GaiaXPInfo

    has_low_glat = True


class SkyMapperMinusGaiaXPOffsetMapMaker(OffsetMapMaker):
    MinuendInfoClass = SkyMapperInfo
    SubtrahendInfoClass = GaiaXPInfo

    has_low_glat = True


class SkyMapperMinusPS1OffsetMapMaker(OffsetMapMaker):
    MinuendInfoClass = SkyMapperInfo
    SubtrahendInfoClass = PS1Info

    has_low_glat = True


class VSTMinusGaiaXPOffsetMapMaker(OffsetMapMaker):
    MinuendInfoClass = VSTInfo
    SubtrahendInfoClass = GaiaXPInfo
