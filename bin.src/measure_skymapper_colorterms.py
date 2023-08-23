#!/usr/bin/env python

from lsst.the.monster import SkyMapperSplineMeasurer


measurer = SkyMapperSplineMeasurer()
measurer.measure_spline_fit(bands=["g", "r", "i", "z"])
