#!/usr/bin/env python

from lsst.the.monster import UBandOffsetMapMaker, SDSSInfo, GaiaXPuInfo

# This is a combined map, DES++ SLR vs SDSS u.  (SDSS footprint)
measurer_sdss = UBandOffsetMapMaker(uband_ref_class=SDSSInfo, apply_offsets=True)
fname = measurer_sdss.measure_uband_offset_map()
measurer_sdss.plot_uband_offset_maps(fname, "uslr-usdss")

# This is a combined map, DES++ SLR vs XP u.  (Full sky)
measurer_xp = UBandOffsetMapMaker(uband_ref_class=GaiaXPuInfo, apply_offsets=True)
fname = measurer_xp.measure_uband_offset_map()
measurer_xp.plot_uband_offset_maps(fname, "uslr-uxp")
