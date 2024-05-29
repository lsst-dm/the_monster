#!/usr/bin/env python

from lsst.the.monster import UBandOffsetMapMaker, SDSSInfo, GaiaXPuInfo


measurer_sdss = UBandOffsetMapMaker(uband_ref_class=SDSSInfo, catalog_info_class_list=[GaiaXPuInfo])
fname = measurer_sdss.measure_uband_offset_map_direct()
measurer_sdss.plot_uband_offset_maps(fname, mode="direct")

measurer = UBandOffsetMapMaker()
fname = measurer.measure_uband_offset_map()
measurer.plot_uband_offset_maps(fname)
