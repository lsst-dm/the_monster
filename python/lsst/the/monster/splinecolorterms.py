import numpy as np
import scipy.optimize

import lsst.afw.math


__all__ = ["ColortermSplineFitter",]


class ColortermSplineFitter:
    """Fit color-terms with splines using median statistics.

    Parameters
    ----------
    mag_color : `np.ndarray`
        The source color (magnitudes).
    flux_target : `np.ndarray`
        The flux array from the reference (target) catalog.
    flux_source : `np.ndarray`
        The flux array from the measured (source) catalog.
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
    def apply_model(pars, mag_color, color_nodes, flux_source=None):
        """Apply the model and compute values.

        Parameters
        ----------
        pars : `np.ndarray`
            Parameters of the polynomial + (optional) flux_offset.
        mag_color : `np.ndarray` (N,)
            Magnitude colors of stars.
        color_nodes : `np.ndarray` (M,)
            Node locations.
        flux_source : `np.ndarray` (N,), optional
            Source flux to apply flux_offset.
        """
        if flux_source is not None:
            has_flux_offset = True
        else:
            has_flux_offset = False

        spl = lsst.afw.math.makeInterpolate(
            color_nodes,
            pars[0: len(color_nodes)],
            lsst.afw.math.stringToInterpStyle("CUBIC_SPLINE"),
        )
        model = spl.interpolate(mag_color)

        if has_flux_offset:
            model -= pars[-1]/flux_source

        return model

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
            flux_source = self._flux_source
        else:
            pars = spline_pars
            flux_source = None

        model = self.apply_model(pars, self._mag_color, self._color_nodes, flux_source=flux_source)

        absdev = np.abs(self._flux_source/self._flux_target - model)
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
        model = self.apply_model(pars, self._mag_color, self._color_nodes, flux_source=self._flux_source)

        absdev = np.abs(self._flux_source/self._flux_target - model)
        t = np.sum(absdev.astype(np.float64))

        return t
