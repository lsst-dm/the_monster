#!/usr/bin/env python

from lsst.the.monster import UBandOffsetMapMaker, SDSSInfo


measurer_sdss = UBandOffsetMapMaker(uband_ref_class=SDSSInfo)
fname = measurer_sdss.measure_uband_offset_direct()
measurer_sdss.plot_uband_offset_maps(fname, mode="direct")

measurer = UBandOffsetMapMaker()
fname = measurer.measure_uband_offset_map()
measurer.plot_uband_offset_maps(fname)
