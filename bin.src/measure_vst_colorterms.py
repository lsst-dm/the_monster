#!/usr/bin/env python

from lsst.the.monster import VSTSplineMeasurer


measurer = VSTSplineMeasurer()
measurer.measure_spline_fit(bands=["g", "r", "i", "z"])
