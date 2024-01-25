#!/usr/bin/env python

from lsst.the.monster import UbandOffsetMapMaker


measurer = UbandOffsetMapMaker()
fname = measurer.measure_uband_offset_map()
measurer.plot_uband_offset_maps(fname)
