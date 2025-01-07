#!/usr/bin/env python

from lsst.the.monster import ComCamSplineMeasurer

measurer = ComCamSplineMeasurer()
measurer.measure_spline_fit(bands=["u", "g", "r", "i", "z", "y"])
