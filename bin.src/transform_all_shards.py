#!/usr/bin/env python

from lsst.the.monster import DESInfo, GaiaDR3Info, MatchAndTransform
import glob


GaiaDR3CatInfoClass = GaiaDR3Info
gaia_info = GaiaDR3CatInfoClass()

# Make a list of all the Gaia shards:
fits_list = sorted(glob.glob(gaia_info.path+'/*.fits'))

# Extract just the htmid from these
gaia_htmids = []
for fits in fits_list:
    tmp = fits.split('/')[-1]
    gaia_htmids.append(tmp.split('.')[0])

# The following line is for running the DES-to-DES only:
mt = MatchAndTransform(catalog_info_class_list=[DESInfo])
# mt = MatchAndTransform()
for htmid in gaia_htmids:
    mt.run(htmid=htmid, verbose=True)
