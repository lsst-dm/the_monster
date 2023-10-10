#!/usr/bin/env python

from lsst.the.monster import DESSplineMeasurer


# This should produce outputs with no change!
measurer = DESSplineMeasurer()
measurer.measure_spline_fit(bands=["g", "r", "i", "z", "y"])
