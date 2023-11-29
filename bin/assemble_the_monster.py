#!/usr/bin/env python

from lsst.the.monster import GaiaDR3Info, AssembleMonsterRefcat
import glob


GaiaDR3CatInfoClass = GaiaDR3Info
gaia_info = GaiaDR3CatInfoClass()

# Make a list of all the Gaia shards:
fits_list = glob.glob(gaia_info.path+'/*.fits')

# Extract just the htmid from these
gaia_htmids = []
for fits in fits_list:
    tmp = fits.split('/')[-1]
    gaia_htmids.append(tmp.split('.')[0])

build_the_monster = AssembleMonsterRefcat()
for htmid in gaia_htmids:
    build_the_monster.run(htmid=htmid)
