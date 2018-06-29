

# needed packages
import numpy as np
# import matplotlib
import matplotlib.pyplot as plt
# import re
from astropy.io import fits
import pyds9
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize

# import needed functions from the toolbox
from ccd_tools import bias_subtract, background_subtract, get_regions


# show the ds9 target
print('ds9 target instance')
print(pyds9.ds9_targets())

# create a DS9 object
ds9 = pyds9.DS9()

# import a list of region definitions
selected_regions = get_regions(get_data=False)

# import the current fits file loaded in DS9
hdu = ds9.get_pyfits()

hdu.info()

# get the bias subtracted data
bias_subtracted_data = bias_subtract(hdu[0])

# use the regions to produce apertures of the data
# also background subtract the data
aperture_list = []  # list for holding aperture data
for region in selected_regions:
    current_aperture = bias_subtracted_data[region.ymin:region.ymax, region.xmin:region.xmax]
    aperture_list.append(current_aperture)


"""calculate the centroid for each aperture"""

# curve function to be fitted
def Gaussian_2d(indata, amplitude, x0, y0, sigma_x, sigma_y, offset):
    """Define gaussian function, assuming no correlation between x and y.

    Uses a flattened input, and gives a flattened output

    Parameters
    ----------
    indata: array int
        indata is a pair of arrays, each array corresponding to the x indice or y indice, in the form (x, y)
    amplitude: float
        represents the total flux of the object being fitted
    x0: float
        horizontal center of the object
    y0: float
        vertical center of the object
    sigma_x: float
        half width half maximum of the object along the horizontal
    sigma_y: float
        half width half maximum of the object along the vertical
    offset: float
        represents the background around the object
    """
    import numpy as np
    x, y = indata
    normalize = 1 / (sigma_x * sigma_y * 2 * np.pi)

    gaussian_fun = offset + amplitude * normalize * np.exp(
        -(x - x0) ** 2 / (2 * sigma_x ** 2) - (y - y0) ** 2 / (2 * sigma_y ** 2))

    return gaussian_fun.ravel()

# generate each curve fit
from scipy.optimize import curve_fit
for aperture in aperture_list:

    print('---------------------')
    # background subtract the aperture
    aperture, mask = background_subtract(aperture)

    # plot the aperture and mask used to background subtract
    norm = ImageNormalize(stretch=SqrtStretch())
    f1, axisarg = plt.subplots(3, 1)
    axisarg[0].imshow(aperture, norm=norm, origin='lower', cmap='viridis')
    axisarg[1].imshow(mask, origin='lower', cmap='viridis')
    axisarg[2].hist(aperture.flatten(),bins=500, range=[-500, 5000])

    # generate a best guess
    x_guess = aperture.shape[0] / 2
    y_guess = aperture.shape[1] / 2
    amp_guess = np.amax(aperture)

    # indexes of the apature, remembering that python indexes vert, horz
    y = np.arange(aperture.shape[0])
    x = np.arange(aperture.shape[1])
    x, y = np.meshgrid(x, y)


    # curve fit
    try:
        g_fit, g_cov = curve_fit(Gaussian_2d, (x, y), aperture.ravel(), p0=[amp_guess, x_guess, y_guess, 1, 1, 1])


    except RuntimeError:
        print('Unable to find fit.')
    else:
        print('Resultant parameters')
        print(g_fit)

        error = np.sqrt(np.diag(g_cov))
        print('Error on parameters')
        print(error)

        x_center = g_fit[1]
        y_center = g_fit[2]
        x_width = g_fit[3]
        y_width = g_fit[4]

        # add the calculated center and width bars to the aperture plot as a cross
        # The width of the lines correspond to the width in that direction
        axisarg[0].errorbar(x_center, y_center, xerr=x_width, yerr=y_width, ecolor='red')
        """Chi squared calculations"""
        observed = aperture.ravel()

        # define the inputs for the 2d gaussian
        g_input = (x, y)
        amplitude = g_fit[0]
        x0 = g_fit[1]
        y0 = g_fit[2]
        sigma_x = g_fit[3]
        sigma_y = g_fit[4]
        offset = g_fit[5]

        expected = Gaussian_2d(g_input, amplitude, x0, y0, sigma_x, sigma_y, offset)

        # calculated raw chi squared
        chisq = sum(np.divide((observed - expected) ** 2, expected))

        # degrees of freedom, 5 parameters
        degrees_of_freedom = observed.size - 5

        # normalized chi squared
        chisq_norm = chisq / degrees_of_freedom

        print('Normalized chi squared:')
        print(chisq_norm)

plt.show()


