#!/usr/bin/env python

from lsst.the.monster import UBandOffsetMapMaker


measurer = UBandOffsetMapMaker()
fname = measurer.measure_uband_offset_map()
measurer.plot_uband_offset_maps(fname)
