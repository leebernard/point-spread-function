# needed packages
import numpy as np
# import matplotlib
import matplotlib.pyplot as plt
# import re
from astropy.io import fits
import pyds9

# import needed functions from the toolbox
from ccd_tools import bias_subtract, background_subtract, parse_regions

# show the ds9 target
print('ds9 target instance')
print(pyds9.ds9_targets())

# create a DS9 object
ds9 = pyds9.DS9()

# import a list of region definitions
selected_regions = parse_regions(get_data=False)

# import the current fits file loaded in DS9
hdu = ds9.get_pyfits()

hdu.info()


# get the bias subtracted data
bias_subtraced_data = bias_subtract(hdu[0])

# slice the data into the specified regions
for region in selected_regions

