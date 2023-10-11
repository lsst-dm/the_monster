import os
import numpy as np
import scipy.optimize
from astropy import units
import yaml

import lsst.afw.math


__all__ = ["ColortermSplineFitter", "MagSplineFitter", "ColortermSpline"]


class ColortermSplineFitter:
    """Fit color-terms with splines using median statistics.

    The terms here will make the fluxes in the source catalog look
    like the target catalog over a wide range of colors.  E.g., make
    PS1 (a source catalog) look like DES (the target catalog, used
    for the intermediate representation.)

    Parameters
    ----------
    mag_color : `np.ndarray`
        The source color (magnitudes).  These should be computed from the
        source catalog.
    flux_target : `np.ndarray`
        The flux array from the target catalog.
    flux_source : `np.ndarray`
        The flux array from the source catalog.
    color_nodes : `np.ndarray`
        Nodes to do color spline fit.
    fit_flux_offset : `bool`, optional
        Simultaneously fit flux offset term?
    """
    def __init__(
            self,
            mag_color,
            flux_target,
            flux_source,
            color_nodes,
            fit_flux_offset=False,
    ):
        self._mag_color = mag_color
        self._flux_target = flux_target
        self._flux_source = flux_source

        self._color_nodes = color_nodes
        self._fit_flux_offset = fit_flux_offset

    def estimate_p0(self):
        """Estimate the initial fit parameters.

        Returns
        -------
        p0 : `np.ndarray`
            Estimate of initial fit parameters.
        """
        npt = len(self._color_nodes)
        if self._fit_flux_offset:
            npt += 1

        p0 = np.zeros(npt)
        p0[0: len(self._color_nodes)] = np.median(self._flux_source/self._flux_target)

        return p0

    @staticmethod
    def apply_model(pars, flux_source, mag_color, color_nodes):
        """Apply the model and compute values.

        If ``pars`` is the same length as ``color_nodes``, then there is
        no flux offset term.  If it is one longer, then the final element
        in ``pars`` is the flux offset term.

        Parameters
        ----------
        pars : `np.ndarray` (M,)
            Parameters of the spline + flux_offset.
        flux_source : `np.ndarray` (N,)
            Source flux values.
        mag_color : `np.ndarray` (N,)
            Magnitude colors of stars.
        color_nodes : `np.ndarray` (M,)
            Node locations.

        Returns
        -------
        flux_model : `np.ndarray` (N,)
            Model flux, after applying color terms to flux_source.
        """
        spl = lsst.afw.math.makeInterpolate(
            color_nodes,
            pars[0: len(color_nodes)],
            lsst.afw.math.stringToInterpStyle("CUBIC_SPLINE"),
        )
        flux_model = flux_source * np.array(spl.interpolate(mag_color))

        # Apply flux offset if there is an extra parameter.
        if len(pars) == (len(color_nodes) + 1):
            flux_model -= pars[-1]

        return flux_model

    def fit(self, p0, n_iter_flux_offset=3):
        """Perform a spline fit, perhaps with flux offset.

        Parameters
        ----------
        p0 : `np.ndarray`
            Initial fit parameters, need to specify org.
        n_iter_flux_offset : `int`, optional
            Number of iterations to fit (if using flux_offset).

        Returns
        -------
        pars : `np.ndarray`
            Best-fit parameters.
        """
        if self._fit_flux_offset:
            n_iter = n_iter_flux_offset
            spline_pars = p0[:-1]
            flux_offset_par = np.array([p0[-1]])
            self._flux_offset_par = flux_offset_par
        else:
            n_iter = 1
            spline_pars = p0

        for i in range(n_iter):
            res = scipy.optimize.minimize(
                self.compute_cost_spline,
                spline_pars,
                method="L-BFGS-B",
                jac=False,
                options={
                    "maxfun": 2000,
                    "maxiter": 2000,
                    "maxcor": 20,
                    "eps": 1e-3,
                    "ftol": 1e-15,
                    "gtol": 1e-15,
                },
            )

            spline_pars = res.x

            if self._fit_flux_offset:
                self._spline_pars = spline_pars

                res = scipy.optimize.minimize(
                    self.compute_cost_flux_offset,
                    flux_offset_par,
                    method="L-BFGS-B",
                    jac=False,
                    options={
                        "maxfun": 2000,
                        "maxiter": 2000,
                        "maxcor": 20,
                        "eps": 1e-3,
                        "ftol": 1e-15,
                        "gtol": 1e-15,
                    },
                )

                flux_offset_par = res.x
                self._flux_offset_par = flux_offset_par

        if not self._fit_flux_offset:
            pars = spline_pars
        else:
            pars = np.concatenate([spline_pars, flux_offset_par])

        return pars

    def compute_cost_spline(self, spline_pars):
        """Compute the median cost function for spline(pars).

        Parameters
        ----------
        spline_pars : `np.ndarray`
            Fit parameters.

        Returns
        -------
        t : `float`
            Median cost.
        """
        if self._fit_flux_offset:
            pars = np.concatenate([spline_pars, self._flux_offset_par])
        else:
            pars = spline_pars

        flux_model = self.apply_model(pars, self._flux_source, self._mag_color, self._color_nodes)

        absdev = np.abs(self._flux_target - flux_model)
        t = np.sum(absdev.astype(np.float64))

        return t

    def compute_cost_flux_offset(self, flux_offset_par):
        """Compute the median cost function for the flux offset.

        Parameters
        ----------
        flux_offset_par : `np.ndarray` (1,)
            Flux offset parameter.

        Returns
        -------
        t : `float`
            Median cost.
        """
        pars = np.concatenate([self._spline_pars, flux_offset_par])

        flux_model = self.apply_model(pars, self._flux_source, self._mag_color, self._color_nodes)

        absdev = np.abs(self._flux_target - flux_model)
        t = np.sum(absdev.astype(np.float64))

        return t


class MagSplineFitter:
    """Fit flux offsets with splines as a function of magnitude using
    median statistics.

    The terms here can be run as an afterburner to correct source fluxes
    to look like target fluxes over a wide range of magnitudes.  E.g.,
    make PS1 (a source catalog) look like XP (the target catalog) up to
    bright magnitudes.

    Parameters
    ----------
    flux_target : `np.ndarray`
        The flux array from the target catalog.
    flux_source : `np.ndarray`
        The flux array from the source catalog. These will be converted
        to AB magnitudes (assuming nJy input).
    mag_nodes : `np.ndarray`
        Nodes to do magnitude offset fit.
    """
    def __init__(self, flux_target, flux_source, mag_nodes):
        self._flux_target = flux_target
        self._flux_source = flux_source
        self._mag_nodes = mag_nodes

    def estimate_p0(self):
        """Estimate the initial fit parameters.

        Returns
        -------
        p0 : `np.ndarray`
            Estimate of initial fit parameters.
        """
        npt = len(self._mag_nodes)

        p0 = np.zeros(npt)
        p0[:] = np.median(self._flux_source/self._flux_target)

        return p0

    @staticmethod
    def apply_model(pars, flux_source, mag_nodes):
        """Apply the model and compute values.

        Parameters
        ----------
        pars : `np.ndarray` (M,)
            Parameters of the spline.
        flux_source : `np.ndarray` (N,)
            Source flux values.
        mag_nodes : `np.ndarray` (M,)
            Node locations.

        Returns
        -------
        flux_model : `np.ndarray` (N,)
            Model flux, after applying flux offsets to flux_source.
        """
        spl = lsst.afw.math.makeInterpolate(
            mag_nodes,
            pars,
            lsst.afw.math.stringToInterpStyle("CUBIC_SPLINE"),
        )
        mag_source = (flux_source*units.nJy).to_value(units.ABmag)

        flux_model = flux_source * np.array(spl.interpolate(mag_source))

        return flux_model

    def fit(self, p0):
        """Perform the spline fit.

        Parameters
        ----------
        p0 : `np.ndarray`
            Initial fit parameters.

        Returns
        -------
        pars : `np.ndarray`
            Best-fit parameters.
        """
        res = scipy.optimize.minimize(
            self.compute_cost_spline,
            p0,
            method="L-BFGS-B",
            jac=False,
            options={
                "maxfun": 2000,
                "maxiter": 2000,
                "maxcor": 20,
                "eps": 1e-3,
                "ftol": 1e-15,
                "gtol": 1e-15,
            },
        )
        pars = res.x

        return pars

    def compute_cost_spline(self, spline_pars):
        """Compute the median cost function for spline(pars).

        Parameters
        ----------
        spline_pars : `np.ndarray`
            Fit parameters.

        Returns
        -------
        t : `float`
            Median cost.
        """
        flux_model = self.apply_model(spline_pars, self._flux_source, self._mag_nodes)

        absdev = np.abs(self._flux_target - flux_model)
        t = np.sum(absdev.astype(np.float64))

        return t


class ColortermSpline:
    """Save, load, and apply spline color terms.

    Parameters
    ----------
    source_survey : `str`
        Name of the source survey (e.g. ``ps1``).
    target_survey : `str`
        Name of the target survey (e.g. ``des``).
    source_color_field_1 : `str`
        Name of first flux field for colors in source survey.
    source_color_field_2 : `str`
        Name of second flux field for colors in source survey.
    source_field : `str`
        Name of flux field to convert to target survey.
    nodes : `np.ndarray` (N,)
        Array of spline nodes.
    spline_values : `np.ndarray` (N,)
        Array of spline values.
    flux_offset : `float`, optional
        Flux offset to apply in conversion.
    mag_nodes : `np.ndarray` (M,), optional
        Array of magnitude spline nodes for additional fixups.
    mag_spline_values : `np.ndarray` (M,), optional
        Array of magnitude spline values.
    """
    def __init__(
            self,
            source_survey,
            target_survey,
            source_color_field_1,
            source_color_field_2,
            source_field,
            nodes,
            spline_values,
            flux_offset=0.0,
            mag_nodes=None,
            mag_spline_values=None,

    ):
        self.source_survey = source_survey
        self.target_survey = target_survey
        self.source_color_field_1 = source_color_field_1
        self.source_color_field_2 = source_color_field_2
        self.source_field = source_field
        self.nodes = np.array(nodes)
        self.spline_values = np.array(spline_values)
        self.flux_offset = flux_offset
        self.mag_nodes = mag_nodes
        self.mag_spline_values = mag_spline_values

        self.spline = lsst.afw.math.makeInterpolate(
            self.nodes,
            self.spline_values,
            lsst.afw.math.stringToInterpStyle("CUBIC_SPLINE"),
        )

        if mag_nodes is not None:
            self.mag_spline = lsst.afw.math.makeInterpolate(
                self.mag_nodes,
                self.mag_spline_values,
                lsst.afw.math.stringToInterpStyle("CUBIC_SPLINE"),
            )
        else:
            self.mag_spline = None

    def save(self, yaml_file, overwrite=False):
        """Serialize to a yaml file.

        Parameters
        ----------
        yaml_file : `str`
            Name of yaml file for output.
        overwrite : `bool`, optional
            Overwrite file if it exists?
        """
        yaml_dict = {
            "source_survey": str(self.source_survey),
            "target_survey": str(self.target_survey),
            "source_color_field_1": str(self.source_color_field_1),
            "source_color_field_2": str(self.source_color_field_2),
            "source_field": str(self.source_field),
            "nodes": [float(val) for val in self.nodes],
            "spline_values": [float(val) for val in self.spline_values],
            "flux_offset": float(self.flux_offset),
        }

        if self.mag_nodes is not None:
            yaml_dict["mag_nodes"] = [float(val) for val in self.mag_nodes]
            yaml_dict["mag_spline_values"] = [float(val) for val in self.mag_spline_values]

        serialized = yaml.safe_dump(yaml_dict)

        if os.path.isfile(yaml_file):
            if not overwrite:
                raise OSError(f"{yaml_file} already exists, and overwrite=False.")
            os.remove(yaml_file)

        with open(yaml_file, "w") as fd:
            fd.write(serialized)

    @classmethod
    def load(cls, yaml_file):
        """Load from a yaml file.

        Parameters
        ----------
        yaml_file : `str`
            Name of yaml file for input.
        """
        with open(yaml_file, "rb") as fd:
            data = yaml.safe_load(fd)

        if "mag_nodes" in data:
            mag_nodes = data["mag_nodes"]
            mag_spline_values = data["mag_spline_values"]
        else:
            mag_nodes = None
            mag_spline_values = None

        return cls(
            data["source_survey"],
            data["target_survey"],
            data["source_color_field_1"],
            data["source_color_field_2"],
            data["source_field"],
            data["nodes"],
            data["spline_values"],
            flux_offset=data["flux_offset"],
            mag_nodes=mag_nodes,
            mag_spline_values=mag_spline_values,
        )

    def apply(self, source_color_flux_1, source_color_flux_2, source_flux):
        """Apply the color term spline model.

        Parameters
        ----------
        source_color_flux_1 : `np.ndarray` (N,)
            Array of source fluxes used for color (1).
        source_color_flux_2 : `np.ndarray` (N,)
            Array of source fluxes used for color (2).
        source_flux : `np.ndarray` (N,)
            Array of source fluxes to convert.

        Returns
        -------
        model_flux : `np.ndarray` (N,)
            Array of fluxes converted to target.
        """
        mag_1 = (np.array(source_color_flux_1)*units.nJy).to_value(units.ABmag)
        mag_2 = (np.array(source_color_flux_2)*units.nJy).to_value(units.ABmag)

        mag_color = mag_1 - mag_2

        model_flux = source_flux * np.array(self.spline.interpolate(mag_color))
        model_flux -= self.flux_offset

        # Check that things are in range: colors out of range simply should
        # not be corrected.
        bad = ((mag_color < self.nodes[0]) | (mag_color > self.nodes[-1]))
        model_flux[bad] = np.nan

        # Apply magnitude offsets if necessary.
        if self.mag_spline is not None:
            mag = np.nan_to_num((model_flux*units.nJy).to_value(units.ABmag))

            flux_offset = np.array(self.mag_spline.interpolate(mag))

            # Set the offsets for those out of range to the last node value.
            flux_offset[mag < self.mag_nodes[0]] = self.mag_spline_values[0]
            flux_offset[mag > self.mag_nodes[-1]] = self.mag_spline_values[-1]

            model_flux *= flux_offset

        return model_flux
