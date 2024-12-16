import os
import numpy as np
import matplotlib.pyplot as plt
from smatch import Matcher
import warnings
from astropy import units
from astropy.table import Table
import fitsio
import scipy.interpolate as interpolate
import scipy.integrate as integrate

from lsst.sphgeom import Box, HtmPixelization
from lsst.utils import getPackageDir

from .splinecolorterms import ColortermSplineFitter, ColortermSpline, MagSplineFitter
from .refcats import GaiaXPInfo, GaiaDR3Info, DESInfo, SkyMapperInfo, PS1Info, VSTInfo, SDSSInfo, GaiaXPuInfo
from .refcats import ComCamInfo, MonsterInfo
from .utils import read_stars


__all__ = [
    "SplineMeasurer",
    "GaiaXPSplineMeasurer",
    "SkyMapperSplineMeasurer",
    "PS1SplineMeasurer",
    "VSTSplineMeasurer",
    "DESSplineMeasurer",
    "GaiaXPuSplineMeasurer",
    "ComCamSplineMeasurer",
]


class SplineMeasurer:
    CatInfoClass = None
    TargetCatInfoClass = DESInfo
    GaiaCatInfoClass = GaiaDR3Info

    fit_mag_offsets = False
    MagOffsetCatInfoClass = None

    target_selection_band = "i"
    apply_target_colorterm = False

    testing_mode = False

    use_custom_target_catalog_reader = False
    do_check_c26202_absolute_calibration = False

    def n_nodes(self, band=None):
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

    def custom_target_catalog_reader(self):
        """Specialized reader for calibration stars.

        Returns
        -------
        catalog : `astropy.Table`
            Astropy table catalog.
        """
        raise NotImplementedError("Must be implemented by subclass")

    def measure_spline_fit(
        self,
        bands=["g", "r", "i", "z", "y"],
        do_plots=True,
        overwrite=False,
    ):
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

        target_info = self.TargetCatInfoClass()

        # Read in all the TARGET stars in the region.
        if self.use_custom_target_catalog_reader:
            target_stars = self.custom_target_catalog_reader()
        else:
            target_stars = read_stars(target_info.path, indices, allow_missing=self.testing_mode)

        # Cut to the good stars; use i-band as general reference.
        selected = target_info.select_stars(target_stars, self.target_selection_band)
        target_stars = target_stars[selected]

        # Read in the Gaia stars in the region.
        gaia_info = self.GaiaCatInfoClass()

        gaia_stars = read_stars(gaia_info.path, indices, allow_missing=self.testing_mode)

        # Match these together.
        with Matcher(target_stars["coord_ra"], target_stars["coord_dec"]) as m:
            idx, i1, i2, d = m.query_knn(
                gaia_stars["coord_ra"],
                gaia_stars["coord_dec"],
                distance_upper_bound=0.5/3600.0,
                return_indices=True,
            )

        target_stars = target_stars[i1]

        # Now the actual running.
        cat_info = self.CatInfoClass()

        cat_stars = read_stars(cat_info.path, indices, allow_missing=self.testing_mode)

        with Matcher(target_stars["coord_ra"], target_stars["coord_dec"]) as m:
            idx, i1, i2, d = m.query_knn(
                cat_stars["coord_ra"],
                cat_stars["coord_dec"],
                distance_upper_bound=0.5/3600.,
                return_indices=True,
            )

        target_stars_matched = target_stars[i1]
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

        if self.do_check_c26202_absolute_calibration:
            c26202_absmags = self.compute_target_c26202_magnitudes()
            c26202_cat_index = self.get_c26202_index(cat_stars)

            c26202_message = "C2602 (ComCam)\n"
            c26202_message += "Band CalSpec  Original  Corrected\n"

        yaml_files = []

        for band_index, band in enumerate(bands):
            print(f"Working on transformations from {cat_info.name} to {target_info.name} for {band}")
            mag_color = cat_info.get_mag_colors(cat_stars_matched, band)
            flux_target = target_stars_matched[target_info.get_flux_field(band)]
            flux_cat = cat_stars_matched[cat_info.get_flux_field(band)]

            color_range = cat_info.get_color_range(band)

            nodes = np.linspace(color_range[0], color_range[1], self.n_nodes(band=band))

            selected = cat_info.select_stars(cat_stars_matched, band)
            selected &= target_info.select_stars(target_stars_matched, band)

            if self.apply_target_colorterm:
                # Apply a colorterm to the target flux first.
                raise RuntimeError("I don't think this is used")
                filename = target_info.colorterm_file(band)
                colorterm_spline = ColortermSpline.load(filename)

                band_1, band_2 = cat_info.get_color_bands(band)
                orig_flux = flux_target
                model_flux = colorterm_spline.apply(
                    target_stars_matched[target_info.get_flux_field(band_1)],
                    target_stars_matched[target_info.get_flux_field(band_2)],
                    orig_flux,
                )

                # Overwrite flux_target and update selected
                flux_target[:] = model_flux

                selected &= np.isfinite(flux_target)

            fitter = ColortermSplineFitter(
                mag_color[selected],
                flux_target[selected],
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
                target_info.name,
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
                    model_mag = (model_flux*units.nJy).to_value(units.ABmag)

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
                target_info.name,
                cat_info.get_flux_field(band_1),
                cat_info.get_flux_field(band_2),
                cat_info.get_flux_field(band),
                nodes,
                spline_values,
                flux_offset=flux_offset,
                mag_nodes=mag_nodes,
                mag_spline_values=mag_spline_values,
            )

            if self.do_check_c26202_absolute_calibration:
                # We first apply the color terms.
                flux_target_corr0 = colorterm.apply(
                    np.array([cat_stars[cat_info.get_flux_field(band_1)][c26202_cat_index]]),
                    np.array([cat_stars[cat_info.get_flux_field(band_2)][c26202_cat_index]]),
                    np.array([cat_stars[cat_info.get_flux_field(band)][c26202_cat_index]]),
                )
                mag_target_corr0 = float((flux_target_corr0*units.nJy).to_value(units.ABmag))

                ratio = flux_target_corr0 / c26202_absmags[band_index].to_value(units.nJy)

                # Fix these for the plots.
                flux_target /= ratio

                # Redefine the color term with the new spline values.
                colorterm = ColortermSpline(
                    cat_info.name,
                    target_info.name,
                    cat_info.get_flux_field(band_1),
                    cat_info.get_flux_field(band_2),
                    cat_info.get_flux_field(band),
                    nodes,
                    spline_values / ratio,
                    flux_offset=flux_offset,
                    mag_nodes=mag_nodes,
                    mag_spline_values=mag_spline_values,
                )

                flux_target_corr1 = colorterm.apply(
                    np.array([cat_stars[cat_info.get_flux_field(band_1)][c26202_cat_index]]),
                    np.array([cat_stars[cat_info.get_flux_field(band_2)][c26202_cat_index]]),
                    np.array([cat_stars[cat_info.get_flux_field(band)][c26202_cat_index]]),
                )
                mag_target_corr1 = float((flux_target_corr1*units.nJy).to_value(units.ABmag))

                c26202_message += f"{band}     "
                c26202_message += f"{c26202_absmags[band_index].value:0.3f}  "
                c26202_message += f"{mag_target_corr0:0.3f}    "
                c26202_message += f"{mag_target_corr1:0.3f}\n"

            yaml_file = f"{cat_info.name}_to_{target_info.name}_band_{band}.yaml"
            colorterm.save(yaml_file, overwrite=overwrite)

            yaml_files.append(yaml_file)

            # Create QA plots if desired.
            if do_plots:
                ratio_extent = np.nanpercentile(flux_cat[selected]/flux_target[selected], [0.5, 99.5])

                xlabel = f"{band_1} - {band_2}"
                ylabel = f"{cat_info.name}_{band}/{target_info.name}_{band}"

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
                plt.title(f"{cat_info.name} {band} color term")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    plt.tight_layout()
                plt.savefig(f"{cat_info.name}_to_{target_info.name}_band_{band}_color_term.png")

                flux_target_corr = colorterm.apply(
                    np.array(cat_stars_matched[cat_info.get_flux_field(band_1)]),
                    np.array(cat_stars_matched[cat_info.get_flux_field(band_2)]),
                    np.array(flux_cat),
                )
                resid = (flux_target_corr[selected] - flux_target[selected])/flux_target[selected]

                resid_extent = np.nanpercentile(resid, [0.5, 99.5])

                xlabel2 = f"mag_{band} ({cat_info.name})"

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
                plt.title(f"{cat_info.name} {band} flux residuals")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    plt.tight_layout()
                plt.savefig(f"{cat_info.name}_to_{target_info.name}_band_{band}_flux_residuals.png")

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

        if self.do_check_c26202_absolute_calibration:
            print(c26202_message)

        return yaml_files

    def get_c26202_index(self, stars):
        """Get the C26202 index for a catalog of stars.

        Returns
        -------
        c26202_index : `int`
            Index of C26202 in the catalog.
        """
        c26202_ra = 15.0*(3 + 32/60. + 32.843/(60.*60.))
        c26202_dec = -1.0*(27 + 51/60. + 48.58/(60.*60.))

        with Matcher(np.asarray(stars["coord_ra"]), np.asarray(stars["coord_dec"])) as m:
            idx, i1, i2, d = m.query_knn(
                [c26202_ra],
                [c26202_dec],
                k=1,
                distance_upper_bound=1.0/3600.,
                return_indices=True,
            )

        if len(i1) == 0:
            raise RuntimeError("Could not find C26202 in catalog.")

        return i1[0]

    def compute_target_c26202_magnitudes(self):
        """Compute C26202 magnitudes for target catalog.

        Returns
        -------
        c26202_abmags : `np.ndarray`
            Array of c26202 AB magnitudes, one for each target band.
        """
        spec_file = os.path.join(
            getPackageDir("the_monster"),
            "data",
            "calspec",
            "c26202_mod_008.fits",
        )
        spec = fitsio.read(spec_file, ext=1, lower=True)

        spec_int_func = interpolate.interp1d(
            spec["wavelength"],
            1.0e23*spec["flux"]*spec["wavelength"]*spec["wavelength"]*1e-10/299792458.0,
        )

        target_info = self.TargetCatInfoClass()

        bands = target_info.bands
        c26202_mags = np.zeros(len(bands))

        throughputs = {}

        if target_info.NAME == "ComCam":
            for band in bands:
                throughput_file = os.path.join(
                    getPackageDir("the_monster"),
                    "data",
                    "throughputs",
                    f"total_comcam_{band}.ecsv",
                )
                throughput = Table.read(throughput_file)

                throughputs[band] = throughput
        else:
            raise NotImplementedError(f"Absolute calibration of C26202 for {target_info.NAME} not supported.")

        for i, band in enumerate(bands):
            throughput = throughputs[band]

            wavelengths = throughput["wavelength"].quantity.to_value(units.Angstrom)

            f_nu = spec_int_func(wavelengths)
            num = integrate.simpson(
                y=f_nu*throughput["throughput"]/wavelengths,
                x=wavelengths,
            )
            denom = integrate.simpson(
                y=throughput["throughput"]/wavelengths,
                x=wavelengths,
            )
            c26202_mags[i] = -2.5*np.log10(num / denom) + 2.5*np.log10(3631)

        c26202_mags *= units.ABmag

        return c26202_mags


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


class GaiaXPuSplineMeasurer(SplineMeasurer):
    # This measurer is used to standardize the Gaia XP "SDSS u" band
    # into calibrated SDSS u, with color corrections based on the
    # g-r color.  Therefore, the target selection is based on the
    # g-band signal-to-noise.

    CatInfoClass = GaiaXPuInfo
    TargetCatInfoClass = SDSSInfo

    target_selection_band = "g"

    def n_nodes(self, band=None):
        return 8

    @property
    def ra_dec_range(self):
        return (20.0, 35.0, -4.0, 4.0)


class ComCamSplineMeasurer(SplineMeasurer):
    # This measurer converts from SDSSu/DESgrizy to ComCam
    # using a previous version of The Monster for simplicity
    # (apologies for the circularity).  It has a custom reader
    # because the data are under embargo.

    CatInfoClass = MonsterInfo
    TargetCatInfoClass = ComCamInfo

    target_selection_band = "r"

    use_custom_target_catalog_reader = True
    do_check_c26202_absolute_calibration = True

    def n_nodes(self, band=None):
        if band in [None, "r", "i", "z"]:
            return 10
        elif band in ["g"]:
            return 8
        elif band in ["y"]:
            return 7
        elif band in ["u"]:
            return 5

        return 10

    def custom_target_catalog_reader(self):
        """Specialized reader for calibration stars from DRP processing
        (embargoed).

        Returns
        -------
        catalog : `astropy.Table`
            Astropy table catalog.
        """
        from lsst.daf.butler import Butler

        butler = Butler(
            "embargo",
            instrument="LSSTComCam",
            collections=["LSSTComCam/runs/DRP/20241101_20241211/w_2024_50/DM-48128"],
        )

        fgcm_stars = butler.get("fgcm_Cycle5_StandardStars")
        md = fgcm_stars.metadata
        fgcm_stars = fgcm_stars.asAstropy()

        fgcm_stars["coord_ra"].convert_unit_to(units.degree)
        fgcm_stars["coord_dec"].convert_unit_to(units.degree)

        stars = Table(
            data={
                "id": fgcm_stars["id"],
                "coord_ra": fgcm_stars["coord_ra"],
                "coord_dec": fgcm_stars["coord_dec"],
            },
        )

        for i, band in enumerate(md.getArray("BANDS")):
            flux = (fgcm_stars["mag_std_noabs"][:, i]*units.ABmag).to_value(units.nJy)
            flux[fgcm_stars["ngood"][:, i] < 2] = np.nan
            flux_err = (np.log(10)/2.5) * np.asarray(fgcm_stars["magErr_std"][:, i]) * flux

            stars[f"comcam_{band}_flux"] = flux*units.nJy
            stars[f"comcam_{band}_fluxErr"] = flux_err*units.nJy

        # This is useful for getting the color ranges to match up consistently.
        self.apply_comcam_c26202_calibration(stars)

        return stars

    def apply_comcam_c26202_calibration(self, stars):
        """Apply C26202 absolute calibration to a catalog.

        Parameters
        ----------
        stars : `astropy.table.Table`
            Catalog to compute absolute calibration for.
        """
        i1 = self.get_c26202_index(stars)

        target_info = self.TargetCatInfoClass()
        bands = target_info.bands

        c26202_mags = self.compute_target_c26202_magnitudes()

        orig_data_mags = np.zeros(len(bands))
        final_data_mags = np.zeros(len(bands))
        for i, band in enumerate(bands):
            orig_data_mags[i] = stars[f"comcam_{band}_flux"][[i1]].quantity.to_value(units.ABmag)[0]

            ratio = stars[f"comcam_{band}_flux"][i1] / c26202_mags[i].to_value(units.nJy)
            stars[f"comcam_{band}_flux"] /= ratio

            final_data_mags[i] = stars[f"comcam_{band}_flux"][[i1]].quantity.to_value(units.ABmag)[0]

        print("C2602 (ComCam)")
        print("Band CalSpec  Original  Corrected")
        for i, band in enumerate(bands):
            print(f"{band}     "
                  f"{c26202_mags[i].value:0.3f}  "
                  f"{orig_data_mags[i]:0.3f}    "
                  f"{final_data_mags[i]:0.3f}")

    @property
    def do_fit_flux_offset(self):
        return False

    @property
    def ra_dec_range(self):
        # This is the ECDFS + EDFS field.
        return (50.0, 60.0, -50.0, -27.0)
