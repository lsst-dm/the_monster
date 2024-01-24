#!/usr/bin/env python

from lsst.the.monster import GaiaXPuSplineMeasurer


measurer = GaiaXPuSplineMeasurer()
measurer.measure_spline_fit(bands=["u"])
