#!/usr/bin/env python

from lsst.the.monster import UBandOffsetMapMaker, SDSSInfo, GaiaXPuInfo


measurer_sdss_direct = UBandOffsetMapMaker(uband_ref_class=SDSSInfo, catalog_info_class_list=[GaiaXPuInfo])
fname = measurer_sdss_direct.measure_uband_offset_map_direct()
measurer_sdss_direct.plot_uband_offset_maps(fname, "uslr-usdss", mode="direct")

measurer_xp = UBandOffsetMapMaker()
fname = measurer_xp.measure_uband_offset_map()
measurer_xp.plot_uband_offset_maps(fname, "uslr-uxp")

measurer_sdss = UBandOffsetMapMaker(uband_ref_class=SDSSInfo)
fname = measurer_sdss.measure_uband_offset_map()
measurer_sdss.plot_uband_offset_maps(fname, "uslr-usdss")
