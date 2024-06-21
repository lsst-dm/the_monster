#!/usr/bin/env python

from lsst.the.monster import UBandOffsetMapMaker, SDSSInfo, GaiaXPuInfo


# This is a direct map, XPu (color corrected) vs SDSS u.
measurer_sdss_direct = UBandOffsetMapMaker(uband_ref_class=SDSSInfo, catalog_info_class_list=[GaiaXPuInfo])
fname = measurer_sdss_direct.measure_uband_offset_map_direct()
measurer_sdss_direct.plot_uband_offset_maps(fname, "uxp-usdss", mode="direct")

# This is a combined map, DES++ SLR vs SDSS u.  (SDSS footprint)
measurer_sdss = UBandOffsetMapMaker(uband_ref_class=SDSSInfo)
fname = measurer_sdss.measure_uband_offset_map()
measurer_sdss.plot_uband_offset_maps(fname, "uslr-usdss")

# This is a combined map, DES++ SLR vs XP u.  (Full sky)
measurer_xp = UBandOffsetMapMaker(uband_ref_class=GaiaXPuInfo)
fname = measurer_xp.measure_uband_offset_map()
measurer_xp.plot_uband_offset_maps(fname, "uslr-uxp")
