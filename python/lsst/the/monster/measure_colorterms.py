import numpy as np
import matplotlib.pyplot as plt
from smatch import Matcher
import warnings
from astropy import units

from lsst.sphgeom import Box, HtmPixelization

from .splinecolorterms import ColortermSplineFitter, ColortermSpline, MagSplineFitter
from .refcats import GaiaXPInfo, GaiaDR3Info, DESInfo, SkyMapperInfo, PS1Info, VSTInfo
from .utils import read_stars


__all__ = [
    "SplineMeasurer",
    "GaiaXPSplineMeasurer",
    "SkyMapperSplineMeasurer",
    "PS1SplineMeasurer",
    "VSTSplineMeasurer",
    "DESSplineMeasurer",
]


class SplineMeasurer:
    CatInfoClass = None
    TargetCatInfoClass = DESInfo
    GaiaCatInfoClass = GaiaDR3Info

    fit_mag_offsets = False
    MagOffsetCatInfoClass = None

    testing_mode = False

    @property
    def n_nodes(self):
        return 10

    @property
    def n_mag_nodes(self):
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
        cat_info = self.CatInfoClass()

        cat_stars = read_stars(cat_info.path, indices, allow_missing=self.testing_mode)

        with Matcher(des_stars["coord_ra"], des_stars["coord_dec"]) as m:
            idx, i1, i2, d = m.query_knn(
                cat_stars["coord_ra"],
                cat_stars["coord_dec"],
                distance_upper_bound=0.5/3600.,
                return_indices=True,
            )

        des_stars_matched = des_stars[i1]
        cat_stars_matched = cat_stars[i2]

        if self.fit_mag_offsets:
            mag_offset_cat_info = self.MagOffsetCatInfoClass()
            mag_offset_cat_stars = read_stars(
                mag_offset_cat_info.path,
                indices,
                allow_missing=self.testing_mode,
            )

            with Matcher(mag_offset_cat_stars["coord_ra"], mag_offset_cat_stars["coord_dec"]) as m:
                idx, i1, i2, d = m.query_knn(
                    gaia_stars["coord_ra"],
                    gaia_stars["coord_dec"],
                    distance_upper_bound=0.5/3600.0,
                    return_indices=True,
                )

            mag_offset_cat_stars = mag_offset_cat_stars[i1]

            with Matcher(mag_offset_cat_stars["coord_ra"], mag_offset_cat_stars["coord_dec"]) as m:
                idx, i1, i2, d = m.query_knn(
                    cat_stars["coord_ra"],
                    cat_stars["coord_dec"],
                    distance_upper_bound=0.5/3600.0,
                    return_indices=True,
                )

            mag_offset_cat_stars_matched2 = mag_offset_cat_stars[i1]
            cat_stars_matched2 = cat_stars[i2]

        yaml_files = []

        for band in bands:
            print(f"Working on transformations from {cat_info.name} to {des_info.name} for {band}")
            mag_color = cat_info.get_mag_colors(cat_stars_matched, band)
            flux_des = des_stars_matched[des_info.get_flux_field(band)]
            flux_cat = cat_stars_matched[cat_info.get_flux_field(band)]

            color_range = cat_info.get_color_range(band)

            nodes = np.linspace(color_range[0], color_range[1], self.n_nodes)

            selected = cat_info.select_stars(cat_stars_matched, band)
            selected &= des_info.select_stars(des_stars_matched, band)

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

            colorterm_init = ColortermSpline(
                cat_info.name,
                des_info.name,
                cat_info.get_flux_field(band_1),
                cat_info.get_flux_field(band_2),
                cat_info.get_flux_field(band),
                nodes,
                spline_values,
                flux_offset=flux_offset,
            )

            mag_nodes = None
            mag_spline_values = None

            if self.fit_mag_offsets:
                print(f"Working on magnitude offsets from {cat_info.name} to "
                      f"{mag_offset_cat_info.name} for {band}")
                # We use the matched2 catalogs to apply color terms and fit
                # any residual magnitude spline offsets. This is primarily for
                # the bright end, comparing PS1 and XP.

                flux_cat2 = cat_stars_matched2[cat_info.get_flux_field(band)]

                # We need to apply the color term, to the matched2 catalog.
                model_flux = colorterm_init.apply(
                    cat_stars_matched2[cat_info.get_flux_field(band_1)],
                    cat_stars_matched2[cat_info.get_flux_field(band_2)],
                    flux_cat2,
                )

                selected2 = cat_info.select_stars(cat_stars_matched2, band)
                model_flux[~selected2] = np.nan

                # Next, we need to apply color terms to the mag offset catalog.
                filename = mag_offset_cat_info.colorterm_file(band)
                mag_offset_colorterm = ColortermSpline.load(filename)

                flux_mag_offset_cat = mag_offset_cat_stars_matched2[mag_offset_cat_info.get_flux_field(band)]
                mag_offset_model_flux = mag_offset_colorterm.apply(
                    mag_offset_cat_stars_matched2[mag_offset_cat_info.get_flux_field(band_1)],
                    mag_offset_cat_stars_matched2[mag_offset_cat_info.get_flux_field(band_2)],
                    flux_mag_offset_cat,
                )

                selected_mag_offset = mag_offset_cat_info.select_stars(mag_offset_cat_stars_matched2, band)
                mag_offset_model_flux[~selected_mag_offset] = np.nan

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model_mag = (model_flux.quantity).to_value(units.ABmag)

                good = (np.isfinite(model_flux) & np.isfinite(mag_offset_model_flux))

                # Magnitude nodes should cover the overlap range of the
                # two catalogs.
                mag_node_range1 = cat_info.get_mag_range(band)
                mag_node_range2 = mag_offset_cat_info.get_mag_range(band)
                mag_node_range = [max(mag_node_range1[0], mag_node_range2[0]),
                                  min(mag_node_range1[1], mag_node_range2[1])]
                # And if both are unlimited we check here.
                if not np.isfinite(mag_node_range[0]):
                    mag_node_range[0] = np.min(model_mag[good])
                if not np.isfinite(mag_node_range[1]):
                    mag_node_range[1] = np.max(model_mag[good])

                mag_nodes = np.linspace(mag_node_range[0], mag_node_range[1], self.n_mag_nodes)

                mag_fitter = MagSplineFitter(
                    np.array(mag_offset_model_flux[good]),
                    np.array(model_flux[good]),
                    mag_nodes,
                )
                mag_p0 = mag_fitter.estimate_p0()
                mag_spline_values = mag_fitter.fit(mag_p0)

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
                mag_nodes=mag_nodes,
                mag_spline_values=mag_spline_values,
            )

            yaml_file = f"{cat_info.name}_to_{des_info.name}_band_{band}.yaml"
            colorterm.save(yaml_file, overwrite=overwrite)

            yaml_files.append(yaml_file)

            # Create QA plots if desired.
            if do_plots:
                ratio_extent = np.nanpercentile(flux_cat[selected]/flux_des[selected], [0.5, 99.5])

                xlabel = f"{band_1} - {band_2}"
                ylabel = f"{cat_info.name}_{band}/{des_info.name}_{band}"

                xvals = np.linspace(color_range[0], color_range[1], 1000)
                yvals = 1./np.array(colorterm.spline.interpolate(xvals))

                plt.clf()
                plt.hexbin(
                    mag_color[selected],
                    (flux_cat[selected] - flux_offset)/flux_des[selected],
                    bins="log",
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

                flux_target_corr = colorterm.apply(
                    np.array(cat_stars_matched[cat_info.get_flux_field(band_1)]),
                    np.array(cat_stars_matched[cat_info.get_flux_field(band_2)]),
                    np.array(flux_cat),
                )
                resid = (flux_target_corr[selected] - flux_des[selected])/flux_des[selected]

                resid_extent = np.nanpercentile(resid, [0.5, 99.5])

                xlabel2 = f"mag_{band} ({cat_info.name})"

                mag_des_selected = (np.array(flux_des[selected])*units.nJy).to_value(units.ABmag)
                mag_extent = np.nanpercentile(mag_des_selected, [0.5, 99.5])

                plt.clf()
                plt.hexbin(
                    mag_des_selected,
                    resid,
                    bins='log',
                    extent=[mag_extent[0], mag_extent[1], resid_extent[0], resid_extent[1]],
                )
                plt.plot(mag_extent, [0, 0], 'r:')
                plt.xlabel(xlabel2)
                plt.ylabel(ylabel + " - model")
                plt.title(f"{cat_info.name} {band} flux residuals")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    plt.tight_layout()
                plt.savefig(f"{cat_info.name}_to_{des_info.name}_band_{band}_flux_residuals.png")

                # Additional plots for magnitude offset pars.
                if self.fit_mag_offsets:
                    xvals = np.linspace(mag_nodes[0], mag_nodes[-1], 1000)
                    yvals = 1./np.array(colorterm.mag_spline.interpolate(xvals))

                    ratio_extent = np.nanpercentile(model_flux[good]/mag_offset_model_flux[good], [0.5, 99.5])

                    plt.clf()
                    plt.hexbin(
                        model_mag[good],
                        model_flux[good]/mag_offset_model_flux[good],
                        bins="log",
                        extent=[mag_nodes[0], mag_nodes[-1], ratio_extent[0], ratio_extent[1]],
                    )
                    plt.plot(xvals, yvals, "r-")
                    plt.xlabel(f"{band}")
                    plt.ylabel(f"{cat_info.name}_{band}/{mag_offset_cat_info.name}_{band}")
                    plt.title(f"{cat_info.name} Magnitude Residual Term")
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        plt.tight_layout()
                    plt.savefig(f"{cat_info.name}_vs_{mag_offset_cat_info.name}_band_{band}_mag_offset.png")

        return yaml_files


class GaiaXPSplineMeasurer(SplineMeasurer):
    CatInfoClass = GaiaXPInfo


class SkyMapperSplineMeasurer(SplineMeasurer):
    CatInfoClass = SkyMapperInfo


class PS1SplineMeasurer(SplineMeasurer):
    CatInfoClass = PS1Info

    fit_mag_offsets = True
    MagOffsetCatInfoClass = GaiaXPInfo


class VSTSplineMeasurer(SplineMeasurer):
    CatInfoClass = VSTInfo


class DESSplineMeasurer(SplineMeasurer):
    CatInfoClass = DESInfo
