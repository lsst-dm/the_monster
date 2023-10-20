#!/usr/bin/env python

from lsst.the.monster import (
    GaiaXPMinusDESOffsetMapMaker,
    PS1MinusDESOffsetMapMaker,
    SkyMapperMinusDESOffsetMapMaker,
    VSTMinusDESOffsetMapMaker,
    PS1MinusGaiaXPOffsetMapMaker,
    SkyMapperMinusGaiaXPOffsetMapMaker,
    SkyMapperMinusPS1OffsetMapMaker,
    VSTMinusGaiaXPOffsetMapMaker,
)


for maker_class in (
        GaiaXPMinusDESOffsetMapMaker,
        PS1MinusDESOffsetMapMaker,
        SkyMapperMinusDESOffsetMapMaker,
        VSTMinusDESOffsetMapMaker,
        PS1MinusGaiaXPOffsetMapMaker,
        SkyMapperMinusGaiaXPOffsetMapMaker,
        SkyMapperMinusPS1OffsetMapMaker,
        VSTMinusGaiaXPOffsetMapMaker,
):
    maker = maker_class()
    hsp_file = maker.measure_offset_map()
    maker.plot_offset_maps(hsp_file)
