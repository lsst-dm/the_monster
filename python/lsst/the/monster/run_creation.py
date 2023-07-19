from create_isolated_cat import *
from random import sample
from glob import glob

if __name__ == "__main__":
    config = CreateAndMatchIsolatedGaiaCat.ConfigClass()
    config_file = "../../../configs/CreateAndMatchIsolatedGaiaCat.config"
    config.load(config_file)

    files = glob(config.primary_catalog_config.path + "*.fits")
    htmid_list = [i[i.rfind("/") + 1 : -5] for i in files]
    nfiles = 100
    htmid_list = sample(htmid_list, nfiles)

    create_isolated_gaia_cat = CreateAndMatchIsolatedGaiaCat(config_file)
    for htmid in htmid_list:
        create_isolated_gaia_cat.run(htmid=htmid)
