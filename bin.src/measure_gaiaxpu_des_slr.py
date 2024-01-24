#!/usr/bin/env python

from lsst.the.monster import GaiaXPuDESSLRSplineMeasurer


measurer = GaiaXPuDESSLRSplineMeasurer()
measurer.measure_spline_fit(bands=["u"])
