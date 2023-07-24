import numpy as np
from astropy.io import ascii
from scipy import interpolate
import json


def write_coeffs():
    dict_coeffs = {"catalog_name": "sdss_16_standards",
                   "band": "g",
                   "coeffs": list(np.ones(13))}
    # print(dict_coeffs)
    with open("transform_coeffs.json", "w") as fp:
        json.dump(dict_coeffs, fp)  # encode dict into JSON
    print("Done writing dict into .json file")

    ## To read this file:
    ## Open the file for reading
    #with open("transform_coeffs.json", "r") as fp:
    #    # Load the dictionary from the file
    #    dict_coeffs = json.load(fp)


def get_transform(inp_id=None):
    '''
    Inputs
    ======
    inp_id: `string`
        Identifier for which survey the input catalog came from. This is used
        as the key to look up transformations.
    Outputs
    =======
    transform_coeffs: `array` of `floats`
        Transformation coefficients used to convert to DESY6 system.
    '''
    ## Open the file for reading
    inp_file = 'transform_coeffs.json'
    with open(inp_file, "r") as fp:
        # Load the dictionary from the file
        dict_coeffs = json.load(fp)

    # I don't think the dict approach works. Wait to see what format Eli
    # persists things in...
 
    # Need to figure out how to identify which entry to select from inp file.
    # cat_coeffs = ascii.read(inp_file)
    # Need to loop over bands (and select separate bands from table).
    return cat_coeffs


def transform_to_DESY6(cat, inp_id):
    '''
    Inputs
    ======
    cat: `array` of `floats`
        Input catalog containing magnitudes to transform to the DESY6 system.
    inp_id: `string`
        Identifier for which survey the input catalog came from. This is used
        as the key to look up transformations.
    Outputs
    =======
    transformed_cat: `array` of `floats`
        Output catalog containing the columns of the input, plus additional
        columns with magnitudes transformed to the DESY6 system.
    '''
    coeffs = get_transform(inp_id)

    # Not sure how to accept coefficients instead of the input xy values.
    f_spline = interpolate.CubicSpline(x_in, y_in)

    # How do we get the list of mag columns?
    for mag_column in mag_columns_list:
        x_interp = np.linspace(np.min(cat[mag_column])-0.1,
                               np.max(cat[mag_column]),
                               1000
                               )
        y_interp = f_spline(x_interp)
        # Now add the transformed fluxes to the original cat as a new column.


    return transformed_cat