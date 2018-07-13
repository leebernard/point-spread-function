

# def Moffat(indata, flux, x0, y0, alpha, beta, offset):
#     """Model of PSF using a single Moffat distribution
#     """
#     x, y = indata
#     normalize = (beta - 1) / (np.pi * alpha ** 2)
#
#     moffat_fun = offset + flux*normalize*(1 + ((x - x0)**2 + (y - y0)**2) / (alpha**2))**(-beta)
#
#     return moffat_fun

#
# def flat_Moffat(indata, flux, x0, y0, alpha, beta, offset):
#     """Model of PSF using a single Moffat distribution
#
#     This function flattens the output, for curve fitting
#     """
#     x, y = indata
#     normalize = (beta-1)/(np.pi*alpha**2)
#
#     moffat_fun = offset + flux*normalize*(1 + ((x-x0)**2 + (y-y0)**2)/(alpha**2))**(-beta)
#
#     return moffat_fun.ravel()


def elliptical_Moffat(indata, flux, x0, y0, beta, a, b, theta, offset):
    """Model of PSF using a single Moffat distribution, with elliptical parameters.

    Includes a parameter for axial alignment.

    """
    x_in, y_in = indata
    # normalize = 1  # (beta - 1) / ((a*b) * np.pi)

    # moffat_fun = offset + flux * normalize * (1 + ((x - x0)**2/a**2 + (y - y0)**2/b**2))**(-beta)
    A = np.cos(theta) ** 2 / a ** 2 + np.sin(theta) ** 2 / b ** 2
    B = 2 * np.cos(theta) * np.sin(theta) * (1 / a ** 2 - 1 / b ** 2)
    C = np.sin(theta) ** 2 / a ** 2 + np.cos(theta) ** 2 / b ** 2

    def moffat_fun(x, y): return (1 + A*(x - x0)**2 + B*(x - x0)*(y - y0) + C*(y - y0)**2)**(-beta)

    # numerical normalization
    # scale steps according to the size of the array.

    x_final = np.amax(x_in) + 20
    y_final = np.amax(y_in) + 20
    x_start = np.amin(x_in) - 20
    y_start = np.amin(y_in) - 20
    # delta_x = .1
    # delta_y = .1

    h = 500
    k = 500

    delta_x = (x_final-x_start)/h
    delta_y = (y_final-y_start)/k

    # create a grid of x and y inputs
    x_step, y_step = np.meshgrid(np.arange(x_start + delta_x/2, x_final + delta_x/2, delta_x), np.arange(y_start + delta_y/2, y_final + delta_y/2, delta_y))

    # sum up the function evaluated at the steps, and multiply by the area of each step
    normalize = np.sum(moffat_fun(x_step, y_step))*delta_x*delta_y

    # forget that, just integrate it
    # normalize, norm_err = dblquad(moffat_fun, -np.inf, np.inf, lambda x: -np.inf, lambda x: np.inf)

    output = offset + flux*moffat_fun(x_in, y_in)/normalize

    return output


def flat_elliptical_Moffat(indata, flux, x0, y0, beta, a, b, theta, offset):
    """Model of PSF using a single Moffat distribution, with elliptical parameters.

    Includes a parameter for  axial alignment. This function flattens the output, for curve fitting.

    """
    x_in, y_in = indata
    # normalize = 1  # (beta - 1) / ((a*b) * np.pi)

    # moffat_fun = offset + flux * normalize * (1 + ((x - x0)**2/a**2 + (y - y0)**2/b**2))**(-beta)
    A = np.cos(theta) ** 2 / a ** 2 + np.sin(theta) ** 2 / b ** 2
    B = 2 * np.cos(theta) * np.sin(theta) * (1 / a ** 2 - 1 / b ** 2)
    C = np.sin(theta) ** 2 / a ** 2 + np.cos(theta) ** 2 / b ** 2

    def moffat_fun(x, y): return (1 + A*(x - x0)**2 + B*(x - x0)*(y - y0) + C*(y - y0)**2)**(-beta)

    # numerical normalization
    # scale steps according to the size of the array.

    x_final = np.amax(x_in) + 20
    y_final = np.amax(y_in) + 20
    x_start = np.amin(x_in) - 20
    y_start = np.amin(y_in) - 20
    # delta_x = .1
    # delta_y = .1

    h = 500
    k = 500

    delta_x = (x_final-x_start)/h
    delta_y = (y_final-y_start)/k

    # create a grid of x and y inputs
    x_step, y_step = np.meshgrid(np.arange(x_start + delta_x/2, x_final + delta_x/2, delta_x), np.arange(y_start + delta_y/2, y_final + delta_y/2, delta_y))

    # sum up the function evaluated at the steps, and multiply by the area of each step
    normalize = np.sum(moffat_fun(x_step, y_step))*delta_x*delta_y

    # forget that, just integrate it
    # normalize, norm_err = dblquad(moffat_fun, -np.inf, np.inf, lambda x: -np.inf, lambda x: np.inf)

    output = offset + flux*moffat_fun(x_in, y_in)/normalize

    return output.ravel()

# needed packages
import numpy as np
# import matplotlib
import matplotlib.pyplot as plt
# import re
from astropy.io import fits


# import needed functions from the toolbox
from ccd_tools import bias_subtract, background_subtract

# open fits file, best practice
# file_name = '/home/lee/Documents/k4m_160319_101212_ori.fits.fz'
# found a better file
file_name = '/home/lee/Documents/k4m_160531_050920_ori.fits.fz'
with fits.open(file_name) as hdu:
    hdu.info()
    data_im1 = hdu[1].data
    # bias subtraction
    bias_subtracted_im1 = bias_subtract(hdu[1])
    gain = hdu[1].header['GAIN']  # retrieve gain in electrons/count
    readnoise = hdu[1].header['RDNOISE']  # retrieve read noise, in e

# first object
# Centroid detection:


# arbitrarily chosen object, section manually entered
# ymin = 1545
# ymax = 1595
# xmin = 1745
# xmax = 1795
#
# ymin = 460
# ymax = 500
# xmin = 1490
# xmax = 1540

ymin = 1306
ymax = 1356
xmin = 1636
xmax = 1706
large_aperture = bias_subtracted_im1[ymin:ymax, xmin:xmax]

# convert object to electron units from count
large_aperture = large_aperture * gain

# Background subtract the object
large_aperture, mask, background_dev = background_subtract(large_aperture)

# calculate the deviation on each pixel
# defined as the sqrt of the sum of squares of pixel and background deviation
# pixel deviation is defined as sqrt of pixel value in electrons
# background deviation should include any read noise and bias noise


# show an image of the aperture
from astropy.visualization import SqrtStretch

from astropy.visualization.mpl_normalize import ImageNormalize

norm = ImageNormalize(stretch=SqrtStretch())

# plt.figure()
f1, axisarg = plt.subplots(2, 1)

axisarg[0].imshow(large_aperture, norm=norm, origin='lower', cmap='viridis')
axisarg[1].imshow(mask, origin='lower', cmap='viridis')

# plt.show()

# slice this into a smaller aperture
object1_data = large_aperture  # [11:37, 25:51]
object1_dev = np.sqrt(object1_data + background_dev**2)

def moffat_fit(indata, dev=None):
    """wrapper for the moffat fit procedure.

    This fit is rather complicated, so it has been wrapped into a function for convience
    """

    # instead fit data to moffat
    from scipy.optimize import curve_fit

    # indexes of the aperture, remembering that python indexes vert, horz
    y = np.arange(indata.shape[0])
    x = np.arange(indata.shape[1])
    x, y = np.meshgrid(x, y)

    # generate a best guess
    flux_guess = np.amax(indata)
    y_guess = indata.shape[0] / 2
    x_guess = indata.shape[1] / 2
    beta_guess = 5
    a_guess = 2
    b_guess = 2
    theta_guess = 0
    offset_guess = 0

    # create bounds for the fit, in an attempt to keep it from blowing up
    """
    flux_bound = [0, np.inf]
    x_bound = [0, object1_data.shape[1]]
    y_bound = [0, object1_data.shape[0]]
    beta_bound = [1, 20]]
    a_bound = [0.1, np.inf]
    b_bound = [0.1, np.inf]
    theta_bound = 0, np.pi]
    offset_bound = [-np.inf, np.inf]
    """
    # format the bounds
    lower_bounds = [0, 0, 0, 1, 0.1, 0.1, 0, -np.inf]
    upper_bounds = [np.inf, indata.shape[1], indata.shape[0], 20, np.inf, np.inf, np.pi,
                    np.inf]
    bounds = (lower_bounds, upper_bounds)  # bounds set as pair of array-like tuples

    guess = [flux_guess, x_guess, y_guess, beta_guess, a_guess, b_guess, theta_guess, offset_guess]

    # generate parameters for fit
    fit_result, fit_cov = curve_fit(flat_elliptical_Moffat, (x, y), indata.ravel(), p0=guess, bounds=bounds,
                                    sigma=dev.ravel(), method='trf')

    """Chi squared calculations
    """
    observed = indata.ravel()

    m_input = (x, y)
    flux = fit_result[0]
    x0 = fit_result[1]
    y0 = fit_result[2]
    beta = fit_result[3]
    a = fit_result[4]
    b = fit_result[5]
    theta = fit_result[6]
    offset = fit_result[7]

    expected = flat_elliptical_Moffat(m_input, flux, x0, y0, beta, a, b, theta, offset)

    # calculated raw chi squared
    chisq = sum(np.divide((observed - expected) ** 2, expected + background_dev**2))

    # degrees of freedom, 5 parameters
    degrees_of_freedom = observed.size - 6

    # normalized chi squared
    chisq_norm = chisq / degrees_of_freedom

    print('normalized chi squared:')
    print(chisq_norm)
    return fit_result, fit_cov


# do the fit
m_fit, m_cov = moffat_fit(object1_data, dev=object1_dev)

error = np.sqrt(np.diag(m_cov))
print('Resultant parameters')
print(f'Flux: {m_fit[0]:.2f}±{error[0]:.2f}')
print(f'Center (x, y): {m_fit[1]:.2f}±{error[1]:.2f}, {m_fit[2]:.2f}±{error[2]:.2f}')
print(f'beta: {m_fit[3]:.2f}±{error[3]:.2f}')
print(f'x-axis eccentricity: {m_fit[4]:.2f}±{error[4]:.2f}')
print(f'y-axis eccentricity: {m_fit[5]:.2f}±{error[5]:.2f}')
print(f'angle of eccentricity(Radians: {m_fit[6]:.3f}±{error[6]:.3f}')
print(f'background: {m_fit[7]:.2f}±{error[7]:.2f}')




# print('Covariance matrix, if that is interesting')
# print(m_cov)

print('peak electron count')
print(np.amax(object1_data))

# display the result as a cross. The width of the lines correspond to the width in that direction
norm = ImageNormalize(stretch=SqrtStretch())

# this is the center for the single moffat
x_center = m_fit[1]
y_center = m_fit[2]
x_width = m_fit[4]  # parameter 'a'
y_width = m_fit[5]  # parameter 'b'

# show the aperture, with the found center displayed
plt.figure()
plt.imshow(object1_data, norm=norm, origin='lower', cmap='viridis')
plt.errorbar(x_center, y_center, xerr=x_width, yerr=y_width, ecolor='red')

# Calculate the resultant 2d spread
y = np.arange(object1_data.shape[0])
x = np.arange(object1_data.shape[1])
x, y = np.meshgrid(x, y)
m_input = (x, y)
m_flux = m_fit[0]
m_x0 = m_fit[1]
m_y0 = m_fit[2]
m_beta = m_fit[3]
m_a = m_fit[4]
m_b = m_fit[5]
m_theta = m_fit[6]
m_offset = m_fit[7]
result = elliptical_Moffat(m_input, m_flux, m_x0, m_y0, m_beta, m_a, m_b, m_theta, m_offset)

# difference between the result and the observed data
result_difference = object1_data - result
# plot the result, compared to the original object
f2, axisarg = plt.subplots(2, 1)
object_im = axisarg[0].imshow(object1_data, norm=norm, origin='lower', cmap='viridis')
residual_im = axisarg[1].imshow(result_difference, norm=norm, origin='lower', cmap='viridis')
f2.colorbar(object_im, ax=axisarg[0])
f2.colorbar(residual_im, ax=axisarg[1])

# histogram
plt.figure()
histogram = plt.hist(object1_data.flatten(),bins=2000, range=[-500, 30000])
#
# plt.figure()
# plt.hist(object1_data.flatten(),bins=2000, range=[-500, 30000])


#ratio between the measured flux, and calculated flux
measured_flux = np.sum(object1_data)
print('measured flux: ' + str(measured_flux))

print('ratio of calculated to measured: ' + str(m_flux/measured_flux))

plt.show() # show all figures
