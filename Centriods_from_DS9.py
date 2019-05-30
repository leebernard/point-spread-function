

# needed packages
import numpy as np
# import matplotlib
import matplotlib.pyplot as plt
import pickle
import os

# import re
from astropy.io import fits
import pyds9
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from scipy.optimize import curve_fit
# from scipy.special import gammaincc
# import needed functions from the toolbox
from ccd_tools import bias_subtract, background_subtract, get_regions

def flat_Gaussian_2d(indata, amplitude, x0, y0, sigma_x, sigma_y, offset):
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


def Gaussian_2d(indata, amplitude, x0, y0, sigma_x, sigma_y, offset):
    """same as Guassian_2D, but does not flatten the result.

    This function is used for producing a 2d array of the result from the fit

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

    return gaussian_fun


def elliptical_Moffat(indata, flux, x0, y0, beta, a, b, theta):
    """Model of PSF using a single Moffat distribution, with elliptical parameters.

    Parameters
    ----------
    indata: list
        a list of 2 arrays. The first array is the x values per data point. The
        second array is the y values per data
        point
    flux: float
        Represents the total flux of the object
    x0: float
        horizontal location of the centroid
    y0: float
        vertical location of the centroid
    beta: float
        change in slope parameter
    a: float
        width parameter in the x direction
    b: float
        width parameter in the y direction
    theta: float
        angle of eccentricity
    offset: float
        estimate of background. Should be zero

    Returns
    -------
    moffat_fun: array-like
        array of data values corresponding to the x and y inputs
    """
    x_in, y_in = indata

    # moffat_fun = offset + flux * normalize * (1 + ((x - x0)**2/a**2 + (y - y0)**2/b**2))**(-beta)
    A = np.cos(theta)**2/a**2 + np.sin(theta)**2/b**2
    B = 2*np.cos(theta)*np.sin(theta)*(1/a**2 - 1/b**2)
    C = np.sin(theta)**2/a**2 + np.cos(theta)**2/b**2

    def moffat_fun(x, y): return (1 + (A*(x - x0)**2 + B*(x - x0)*(y - y0) + C*(y - y0)**2) *(2**(1/beta) - 1))**(-beta)

    # numerical normalization
    # scale steps according to the size of the array.

    x_final = np.amax(x_in) + 20
    y_final = np.amax(y_in) + 20
    x_start = np.amin(x_in) - 20
    y_start = np.amin(y_in) - 20
    # delta_x = .1
    # delta_y = .1

    h = 300
    k = 300

    delta_x = (x_final-x_start)/h
    delta_y = (y_final-y_start)/k

    # create a grid of x and y inputs
    x_step, y_step = np.meshgrid(np.arange(x_start + delta_x/2, x_final + delta_x/2, delta_x), np.arange(y_start + delta_y/2, y_final + delta_y/2, delta_y))

    # sum up the function evaluated at the steps, and multiply by the area of each step
    normalize = np.sum(moffat_fun(x_step, y_step))*delta_x*delta_y
    # normalize = 1

    # forget that, just integrate it
    # normalize, norm_err = dblquad(moffat_fun, -np.inf, np.inf, lambda x: -np.inf, lambda x: np.inf)

    output = flux*moffat_fun(x_in, y_in)/normalize

    return output


def flat_elliptical_Moffat(indata, flux, x0, y0, beta, a, b, theta):
    """Model of PSF using a single Moffat distribution, with elliptical parameters.

    Includes a parameter for  axial alignment. This function flattens the
    output, for curve fitting.

    Parameters
    ----------
    indata: list
        a list of 2 arrays. The first array is the x values per data point. The
        second array is the y values per data point.
    flux: float
        Represents the total flux of the object
    x0: float
        horizontal location of the centroid
    y0: float
        vertical location of the centroid
    beta: float
        change in slope parameter
    a: float
        width parameter in the x direction
    b: float
        width parameter in the y direction
    theta: float
        angle of eccentricity
    offset: float
        estimate of background. Should be zero

    Returns
    -------
    moffat_fun.ravel(): flattened array-like
        array of data values produced from the x and y inputs. Flattened, for
        curve fitting
    """
    x_in, y_in = indata

    # moffat_fun = offset + flux * normalize * (1 + ((x - x0)**2/a**2 + (y - y0)**2/b**2))**(-beta)
    A = np.cos(theta) ** 2 / a ** 2 + np.sin(theta) ** 2 / b ** 2
    B = 2 * np.cos(theta) * np.sin(theta) * (1 / a ** 2 - 1 / b ** 2)
    C = np.sin(theta) ** 2 / a ** 2 + np.cos(theta) ** 2 / b ** 2

    def moffat_fun(x, y): return (1 + (A*(x - x0)**2 + B*(x - x0)*(y - y0) + C*(y - y0)**2)*(2**(1/beta) - 1))**(-beta)

    # numerical normalization
    # scale steps according to the size of the array.

    x_final = np.amax(x_in) + 20
    y_final = np.amax(y_in) + 20
    x_start = np.amin(x_in) - 20
    y_start = np.amin(y_in) - 20
    # delta_x = .1
    # delta_y = .1

    h = 300
    k = 300

    delta_x = (x_final-x_start)/h
    delta_y = (y_final-y_start)/k

    # create a grid of x and y inputs
    x_step, y_step = np.meshgrid(np.arange(x_start + delta_x/2, x_final + delta_x/2, delta_x), np.arange(y_start + delta_y/2, y_final + delta_y/2, delta_y))

    # sum up the function evaluated at the steps, and multiply by the area of each step
    normalize = np.sum(moffat_fun(x_step, y_step))*delta_x*delta_y
    # normalize = 1

    # forget that, just integrate it
    # normalize, norm_err = dblquad(moffat_fun, -np.inf, np.inf, lambda x: -np.inf, lambda x: np.inf)

    output = flux*moffat_fun(x_in, y_in)/normalize

    return output.ravel()



# show the ds9 target
print('ds9 target instance')
print(pyds9.ds9_targets())

# create a DS9 object
ds9 = pyds9.DS9()

# load the region file
# regionfile = '/home/lee/Documents/ds9regions/DECam_N3_B.reg'
# ds9.set('regions load ' + regionfile)
# ds9.set('regions select all')


# import the current fits file loaded in DS9
hdu = ds9.get_pyfits()

hdu.info()

# import a list of region definitions
selected_regions = get_regions(get_data=False)

# sort the regions according to the distance from origin
selected_regions.sort(key=lambda region: np.sqrt(region.x_coord**2 + region.y_coord**2))

# get the bias subtracted data
bias_subtracted_data = bias_subtract(hdu[0], keyword='BIASSECA')
gain = hdu[0].header['GAINA']
print(f'GAIN: {gain}')

# use the regions to produce apertures of thedata
# also background subtract the data
aperture_list = []  # list for holding aperture data
lower_left = []
for region in selected_regions:
    current_aperture = bias_subtracted_data[region.ymin:region.ymax, region.xmin:region.xmax]
    lower_left.append([region.ymin, region.xmin])

    # convert to electron count
    current_aperture = current_aperture*gain
    aperture_list.append(current_aperture)


"""calculate the centroid for each aperture"""


# variables for holding the results
background_results = [] # holds background, dev
aperture_data = []
fit_results = []
fit_cov = []

# generate each curve fit

for n, aperture in enumerate(aperture_list):

    print('---------------------')
    print('Aperture '+str(n))
    # background subtract the aperture
    background_value, mask, background_dev = background_subtract(aperture)

    aperture = aperture - background_value

    # generate the associated pixel error
    aperture_err = np.sqrt(aperture + background_dev**2)

    # plot the aperture and mask used to background subtract
    norm = ImageNormalize(stretch=SqrtStretch())
    f1, axisarg = plt.subplots(2, 2, figsize=(10, 10))
    aperture_im = axisarg[0][0].imshow(aperture, norm=norm, origin='lower', cmap='viridis')
    f1.colorbar(aperture_im, ax=axisarg[0][0])
    axisarg[0][0].set_title('Object, with colorscale intensity')
    mask_im = axisarg[1][0].imshow(mask, origin='lower', cmap='viridis')
    axisarg[1][0].set_title('Mask used in background calculations')
    aperture_hist = axisarg[1][1].hist(aperture.flatten(),bins=500, range=[-100, 5000])
    axisarg[1][1].set_title('Object histogram after background subtraction')

    # create bounds for the fit, in an attempt to keep it from blowing up
    """
    flux_bound = [0, np.inf]
    x_bound = [0, object1_data.shape[1]]
    y_bound = [0, object1_data.shape[0]]
    beta_bound = [1, 20]]
    a_bound = [0.1, np.inf]
    b_bound = [0.1, np.inf]
    theta_bound = 0, np.pi/2]
    offset_bound = [-np.inf, np.inf] REMOVED
    """
    # format the bounds
    lower_bounds = [0, 0, 0, 1, 0.1, 0.1, 0]
    upper_bounds = [np.inf, aperture.shape[1], aperture.shape[0], 20, np.inf, np.inf, np.pi/2]
    bounds = (lower_bounds, upper_bounds)  # bounds set as pair of array-like tuples

    # generate a best guess
    flux_guess = np.amax(aperture)*10
    y_guess = aperture.shape[0]/2
    x_guess = aperture.shape[1]/2
    beta_guess = 2
    a_guess = 2
    b_guess = 2
    theta_guess = 0
    # offset_guess = 0

    guess = [flux_guess, x_guess, y_guess, beta_guess, a_guess, b_guess, theta_guess]

    # indexes of the apature, remembering that python indexes vert, horz
    y = np.arange(aperture.shape[0])
    x = np.arange(aperture.shape[1])
    x, y = np.meshgrid(x, y)


    # curve fit
    try:
        m_fit, m_cov = curve_fit(flat_elliptical_Moffat, (x, y), aperture.ravel(), sigma=aperture_err.ravel(), p0=guess,
                                 bounds=bounds)

    except RuntimeError:
        print('Unable to find fit.')
        axisarg[0][1].set_title('Fit not found within parameter bounds')
        del lower_left[n]  # remove the lower left location that failed
    else:

        error = np.sqrt(np.diag(m_cov))
        # print('Error on parameters')
        # print(error)

        # save the results
        background_results.append([background_value, background_dev])
        aperture_data.append(aperture)
        fit_results.append(m_fit)
        fit_cov.append(m_cov)

        x_center = m_fit[1]
        y_center = m_fit[2]
        x_width = m_fit[4]
        y_width = m_fit[5]

        """calculate the resulting plot from the fit"""

        # define the inputs for the elliptical Moffat
        m_input = (x, y)
        m_flux = m_fit[0]
        m_x0 = m_fit[1]
        m_y0 = m_fit[2]
        m_beta = m_fit[3]
        m_a = m_fit[4]
        m_b = m_fit[5]
        m_theta = m_fit[6]
        # m_offset = m_fit[7]

        result = elliptical_Moffat(m_input, m_flux, m_x0, m_y0, m_beta, m_a, m_b, m_theta)

        # calculate the difference between the obersved and the result fro mthe fit
        result_difference = aperture - result



        """Chi squared calculations"""
        observed = aperture.ravel()

        expected = flat_elliptical_Moffat(m_input, m_flux, m_x0, m_y0, m_beta, m_a, m_b, m_theta)

        # calculated raw chi squared
        chisq = sum(np.divide((observed - expected) ** 2, expected + background_dev**2))

        # degrees of freedom, 5 parameters
        degrees_of_freedom = observed.size - 8

        # normalized chi squared
        chisq_norm = chisq / degrees_of_freedom

        #print the results
        print('Resultant parameters')
        print(f'Flux: {m_flux:.2f}±{error[0]:.2f}')
        print(f'Center (x, y): {m_x0:.2f}±{error[1]:.2f}, {m_y0:.2f}±{error[2]:.2f}')
        print(f'beta: {m_beta:.2f}±{error[3]:.2f}')
        print(f'x-axis eccentricity: {m_a:.2f}±{error[4]:.2f}')
        print(f'y-axis eccentricity: {m_b:.2f}±{error[5]:.2f}')
        print(f'angle of eccentricity(Radians: {m_theta:.3f}±{error[6]:.3f}')
        # print(f'background: {m_offset:.2f}±{error[7]:.2f}')

        print('Normalized chi squared: ')
        print(chisq_norm)

        #plot it!
        residual_im = axisarg[0][1].imshow(result_difference, norm=norm, origin='lower', cmap='viridis')
        f1.colorbar(residual_im, ax=axisarg[0][1])
        axisarg[0][1].set_title(f'Residuals. Chisq={chisq_norm:.2f}')
        # add the calculated center and width bars to the aperture plot as a cross
        # The width of the lines correspond to the width in that direction
        # axisarg[0][0].errorbar(x_center, y_center, xerr=x_width, yerr=y_width, ecolor='red')




# convert lower left values into a numpy array
lower_left = np.asarray(lower_left)

# pack the results as a dictionary
archive = {'apertures': aperture_data, 'background': background_results, 'parameters': fit_results,
           'param_cov': fit_cov, 'location': lower_left}

# routine for saving the aperture data
filename = '/home/lee/Documents/decam-91s-S4-A-archive.pkl'

if os.path.isfile(filename):
    input('File already exists. continue...?')
with open(filename, mode='wb') as file:
    pickle.dump(archive, file)

plt.show()
# Loading procedure
# with open(filename, mode='rb') as file:
#     archive = pickle.load(file)

