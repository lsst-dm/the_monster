import numpy as np
import matplotlib.pyplot as plt
from smatch import Matcher
import warnings

from lsst.sphgeom import Box, HtmPixelization

from .splinecolorterms import ColortermSplineFitter, ColortermSpline
from .refcats import GaiaXPInfo, GaiaDR3Info, DESInfo, SkyMapperInfo, PS1Info, VSTInfo
from .utils import read_stars


__all__ = [
    "SplineMeasurer",
    "GaiaXPSplineMeasurer",
    "SkyMapperSplineMeasurer",
    "PS1SplineMeasurer",
    "VSTSplineMeasurer",
]


class SplineMeasurer:
    CatInfoClass = None
    TargetCatInfoClass = DESInfo
    GaiaCatInfoClass = GaiaDR3Info

    testing_mode = False

    @property
    def n_nodes(self):
        return 10

    @property
    def do_fit_flux_offset(self):
        return True

    @property
    def ra_dec_range(self):
        """Get ra/dec range to do fit.

        Returns
        -------
        ra_min, ra_max, dec_min, dec_max : `float`
        """
        return (45.0, 55.0, -30.0, -20.0)

    def measure_spline_fit(self, bands=["g", "r", "i", "z", "y"], do_plots=True, overwrite=False):
        """Measure the spline fit, and save to a yaml file.

        Parameters
        ----------
        bands : `list` [`str`]
            Name of bands to compute color terms.
        do_plots : `bool`, optional
            Make QA plots.
        overwrite : `bool`, optional
            Overwrite an existing yaml color term file.

        Returns
        -------
        yaml_files : `list` [`str`]
            Name of yaml files with color term for each band.
        """
        ra_min, ra_max, dec_min, dec_max = self.ra_dec_range

        box = Box.fromDegrees(ra_min, dec_min, ra_max, dec_max)

        pixelization = HtmPixelization(7)
        rs = pixelization.envelope(box)

        indices = []
        for (begin, end) in rs:
            indices.extend(range(begin, end))

        des_info = self.TargetCatInfoClass()

        # Read in all the DES stars in the region.
        des_stars = read_stars(des_info.path, indices, allow_missing=self.testing_mode)

        # Cut to the good stars; use i-band as general reference.
        selected = des_info.select_stars(des_stars, "i")
        des_stars = des_stars[selected]

        # Read in the Gaia stars in the region.
        gaia_info = self.GaiaCatInfoClass()

        gaia_stars = read_stars(gaia_info.path, indices, allow_missing=self.testing_mode)

        # Match these together.
        with Matcher(des_stars["coord_ra"], des_stars["coord_dec"]) as m:
            idx, i1, i2, d = m.query_knn(
                gaia_stars["coord_ra"],
                gaia_stars["coord_dec"],
                distance_upper_bound=0.5/3600.0,
                return_indices=True,
            )

        des_stars = des_stars[i1]

        # Now the actual running.
        # cat_info = self.get_cat_info()
        cat_info = self.CatInfoClass()

        cat_stars = read_stars(cat_info.path, indices, allow_missing=self.testing_mode)

        with Matcher(des_stars["coord_ra"], des_stars["coord_dec"]) as m:
            idx, i1, i2, d = m.query_knn(
                cat_stars["coord_ra"],
                cat_stars["coord_dec"],
                distance_upper_bound=0.5/3600.,
                return_indices=True,
            )

        des_stars = des_stars[i1]
        cat_stars = cat_stars[i2]

        yaml_files = []

        for band in bands:
            mag_color = cat_info.get_mag_colors(cat_stars, band)
            flux_des = des_stars[des_info.get_flux_field(band)]
            flux_cat = cat_stars[cat_info.get_flux_field(band)]

            color_range = cat_info.get_color_range(band)

            nodes = np.linspace(color_range[0], color_range[1], self.n_nodes)

            selected = cat_info.select_stars(cat_stars, band)

            fitter = ColortermSplineFitter(
                mag_color[selected],
                flux_des[selected],
                flux_cat[selected],
                nodes,
                fit_flux_offset=self.do_fit_flux_offset,
            )

            p0 = fitter.estimate_p0()
            pars = fitter.fit(p0)

            band_1, band_2 = cat_info.get_color_bands(band)

            if self.do_fit_flux_offset:
                spline_values = pars[:-1]
                flux_offset = pars[-1]
            else:
                spline_values = pars
                flux_offset = 0.0

            # Create an spline object to serialize it.
            colorterm = ColortermSpline(
                cat_info.name,
                des_info.name,
                cat_info.get_flux_field(band_1),
                cat_info.get_flux_field(band_2),
                cat_info.get_flux_field(band),
                nodes,
                spline_values,
                flux_offset=flux_offset,
            )

            yaml_file = f"{cat_info.name}_to_{des_info.name}_band_{band}.yaml"
            colorterm.save(yaml_file, overwrite=overwrite)

            yaml_files.append(yaml_file)

            # Create QA plots if desired.
            if do_plots:
                ratio_extent = np.percentile(flux_cat[selected]/flux_des[selected], [0.5, 99.5])

                xlabel = f"{band_1} - {band_2}"
                ylabel = f"{cat_info.name}_{band}/{des_info.name}_band"

                xvals = np.linspace(color_range[0], color_range[1], 1000)
                yvals = colorterm.spline.interpolate(xvals)

                plt.clf()
                plt.hexbin(
                    mag_color[selected],
                    flux_cat[selected]/flux_des[selected],
                    bins='log',
                    extent=[color_range[0], color_range[1], ratio_extent[0], ratio_extent[1]],
                )
                plt.plot(xvals, yvals, "r-")
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                plt.title(f"{cat_info.name} {band} color term")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    plt.tight_layout()
                plt.savefig(f"{cat_info.name}_to_{des_info.name}_band_{band}_color_term.png")

                model = colorterm.apply(
                    cat_stars[cat_info.get_flux_field(band_1)],
                    cat_stars[cat_info.get_flux_field(band_2)],
                    flux_cat,
                )
                resid = flux_cat[selected]/flux_des[selected] - model[selected]

                resid_extent = np.percentile(resid, [0.5, 99.5])

                xlabel2 = f"log10({cat_info.name} band)"

                flux_extent = np.percentile(np.log10(flux_des[selected]), [0.5, 99.5])

                plt.clf()
                plt.hexbin(
                    np.log10(flux_des[selected]),
                    resid,
                    bins='log',
                    extent=[flux_extent[0], flux_extent[1], resid_extent[0], resid_extent[1]],
                )
                plt.plot(flux_extent, [0, 0], 'r:')
                plt.xlabel(xlabel2)
                plt.ylabel(ylabel + " - model")
                plt.title(f"{cat_info.name} {band} flux residuals")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    plt.tight_layout()
                plt.savefig(f"{cat_info.name}_to_{des_info.name}_band_{band}_flux_residuals.png")

        return yaml_files


class GaiaXPSplineMeasurer(SplineMeasurer):
    CatInfoClass = GaiaXPInfo


class SkyMapperSplineMeasurer(SplineMeasurer):
    CatInfoClass = SkyMapperInfo


class PS1SplineMeasurer(SplineMeasurer):
    CatInfoClass = PS1Info


class VSTSplineMeasurer(SplineMeasurer):
    CatInfoClass = VSTInfo
