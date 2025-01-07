import os
import numpy as np
import matplotlib.pyplot as plt
import importlib.resources
import fitsio
import astropy.table
import scipy.interpolate as interpolate
import scipy.integrate as integrate
import warnings

# import fgcm
import lsst.utils

from .refcats import DESInfo, SynthLSSTInfo
from .splinecolorterms import ColortermSplineFitter, ColortermSpline


__all__ = ["SynthLSSTSplineMeasurer"]


class SynthLSSTSplineMeasurer:
    DESInfoClass = DESInfo
    SynthLSSTInfoClass = SynthLSSTInfo
    THROUGHPUT_PATH = None

    def __init__(self):
        if self.THROUGHPUT_PATH is None:
            self._throughput_path = os.path.join(
                lsst.utils.getPackageDir("the_monster"),
                "data",
                "throughputs",
            )
        else:
            self._throughput_path = self.THROUGHPUT_PATH

    def lsst_throughput_file(self, band):
        """Get the LSST throughput file for this band,

        Parameters
        ----------
        band : `str`

        Returns
        -------
        filename : `str`
        """
        filename = os.path.join(
            self._throughput_path,
            f"total_{band}.dat",
        )

        return filename

    def des_throughput_file(self):
        """Get the DES throughput file.

        Returns
        -------
        filename : `str`
        """
        filename = os.path.join(
            self._throughput_path,
            "scidoc1884.txt.fits",
        )

        return filename

    @property
    def n_nodes(self):
        return 4

    def measure_synth_spline_fit(self, bands=["g", "r", "i", "z", "y"], do_plots=True, overwrite=False):
        """Measure the synthetic spline fit, and save to a yaml file.

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
        des_info = self.DESInfoClass()
        lsst_info = self.SynthLSSTInfoClass()

        # Read the templates.
        # These are the FGCM ingestion of the SDSS + Kurusz templates
        # assembled by Pat Kelly in
        # https://ui.adsabs.harvard.edu/abs/2014MNRAS.439...28K/abstract
        template_file = importlib.resources.files("fgcm.data.templates").joinpath(
            "stellar_templates_master.fits"
        )

        fits = fitsio.FITS(template_file)
        fits.update_hdu_list()
        ext_names = []
        for hdu in fits.hdu_list:
            ext_name = hdu.get_extname()
            if ('TEMPLATE_' in ext_name):
                ext_names.append(ext_name)

        n_templates = len(ext_names)

        templates = {}
        for i in range(n_templates):
            templates[i] = fits[ext_names[i]].read(lower=True)
        fits.close()

        throughputs = {}
        for band in lsst_info.bands:
            tput = astropy.table.Table.read(self.lsst_throughput_file(band), format="ascii")
            tput.rename_column("col1", "wavelength")
            tput.rename_column("col2", "throughput")
            throughputs[band] = tput

        # Make the synthetic LSST catalog.
        dtype = []
        for band in lsst_info.bands:
            flux_field = lsst_info.get_flux_field(band)
            dtype.append((flux_field, "f8"))
            dtype.append((flux_field + "Err", "f8"))

        synth_lsst_cat = np.zeros(n_templates, dtype=dtype)

        for i in range(n_templates):
            for j, band in enumerate(lsst_info.bands):
                template_lambda = templates[i]['lambda']
                template_f_lambda = templates[i]['flux']
                template_f_nu = template_f_lambda * template_lambda * template_lambda

                int_func = interpolate.interp1d(
                    template_lambda,
                    template_f_nu,
                    bounds_error=False,
                    fill_value=(template_f_nu[0], template_f_nu[-1]),
                )
                # The throughputs are defined in nm, convert to Angstrom.
                tput_lambda = throughputs[band]["wavelength"]*10.
                f_nu = int_func(tput_lambda)

                num = integrate.simpson(
                    y=f_nu*throughputs[band]["throughput"]/tput_lambda,
                    x=tput_lambda,
                )
                denom = integrate.simpson(
                    y=throughputs[band]["throughput"]/tput_lambda,
                    x=tput_lambda,
                )

                synth_lsst_cat[lsst_info.get_flux_field(band)][i] = num/denom

        # Now the DES catalog
        des_passbands = astropy.table.Table.read(self.des_throughput_file(), format="fits")

        # Patch in the y -> Y
        ind = des_info.bands.index("y")
        des_info.bands[ind] = "Y"

        dtype = []
        for band in des_info.bands:
            flux_field = des_info.get_flux_field(band)
            dtype.append((flux_field, "f8"))
            dtype.append((flux_field + "Err", "f8"))

        synth_des_cat = np.zeros(n_templates, dtype=dtype)

        for i in range(n_templates):
            for j, band in enumerate(des_info.bands):
                template_lambda = templates[i]['lambda']
                template_f_lambda = templates[i]['flux']
                template_f_nu = template_f_lambda * template_lambda * template_lambda

                int_func = interpolate.interp1d(
                    template_lambda,
                    template_f_nu,
                    bounds_error=False,
                    fill_value=(template_f_nu[0], template_f_nu[-1]),
                )
                tput_lambda = des_passbands["LAMBDA"]
                f_nu = int_func(tput_lambda)

                num = integrate.simpson(
                    y=f_nu*des_passbands[band]/tput_lambda,
                    x=tput_lambda,
                )
                denom = integrate.simpson(
                    y=des_passbands[band]/tput_lambda,
                    x=tput_lambda,
                )

                synth_des_cat[des_info.get_flux_field(band)][i] = num/denom

        yaml_files = []

        for band in bands:
            mag_color = des_info.get_mag_colors(synth_des_cat, band)
            flux_des = synth_des_cat[des_info.get_flux_field(band)].copy()
            flux_lsst = synth_lsst_cat[lsst_info.get_flux_field(band)].copy()

            color_range = lsst_info.get_color_range(band)

            # We will normalize the lsst fluxes over this color range.
            selected = ((mag_color > color_range[0]) & (mag_color < color_range[1]))
            ratio = np.median(flux_lsst[selected]/flux_des[selected])
            flux_lsst /= ratio

            nodes = np.linspace(color_range[0], color_range[1], self.n_nodes)

            fitter = ColortermSplineFitter(
                # Color in source catalog.
                mag_color[selected],
                # flux in the target survey.
                flux_lsst[selected],
                # flux in the source survey.
                flux_des[selected],
                nodes,
            )

            p0 = fitter.estimate_p0()
            pars = fitter.fit(p0)

            band_1, band_2 = des_info.get_color_bands(band)

            spline_values = pars

            colorterm = ColortermSpline(
                des_info.name,
                lsst_info.name,
                des_info.get_flux_field(band_1),
                des_info.get_flux_field(band_2),
                des_info.get_flux_field(band),
                nodes,
                spline_values,
            )

            yaml_file = f"{des_info.name}_to_{lsst_info.name}_band_{band}.yaml"
            colorterm.save(yaml_file, overwrite=overwrite)

            yaml_files.append(yaml_file)

            if do_plots:
                ratio_extent = np.nanpercentile(flux_des[selected]/flux_lsst[selected], [0.5, 99.5])

                xlabel = f"{band_1} - {band_2}"
                ylabel = f"{des_info.name}_{band}/{lsst_info.name}_{band}"

                xvals = np.linspace(color_range[0], color_range[1], 1000)
                yvals = 1./np.array(colorterm.spline.interpolate(xvals))

                plt.clf()
                plt.plot(
                    mag_color[selected],
                    flux_des[selected]/flux_lsst[selected],
                    'k.',
                )
                plt.xlim(color_range[0], color_range[1])
                plt.ylim(ratio_extent[0], ratio_extent[1])
                plt.plot(xvals, yvals, "r-")
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                plt.title(f"{des_info.name} {band} color term")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    plt.tight_layout()
                plt.savefig(f"{des_info.name}_to_{lsst_info.name}_band_{band}_color_term.png")

        return yaml_files
