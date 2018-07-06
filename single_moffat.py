#
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


def elliptical_Moffat(indata, flux, x0, y0, beta, a, b, offset):
    """Model of PSF using a single Moffat distribution, with elliptical parameters.

    This version does not account for axial alignment.

    This function flattens the output, for curve fitting
    """
    x, y = indata
    normalize = (beta - 1) / ((a*b) * np.pi)

    moffat_fun = offset + flux * normalize * (1 + ((x - x0)**2/a**2 + (y - y0)**2/b**2))**(-beta)

    return moffat_fun



def flat_elliptical_Moffat(indata, flux, x0, y0, beta, a, b, offset):
    """Model of PSF using a single Moffat distribution, with elliptical parameters.

    This version does not account for axial alignment

    This function flattens the output, for curve fitting

    """
    x, y = indata
    normalize = (beta - 1) / ((a*b) * np.pi)

    moffat_fun = offset + flux * normalize * (1 + ((x - x0)**2/a**2 + (y - y0)**2/b**2))**(-beta)

    return moffat_fun.ravel()


# needed packages
import numpy as np
# import matplotlib
import matplotlib.pyplot as plt
# import re
from astropy.io import fits


# import needed functions from the toolbox
from ccd_tools import bias_subtract, background_subtract

# open fits file, best practice
file_name = '/home/lee/Documents/k4m_160319_101212_ori.fits.fz'
with fits.open(file_name) as hdu:
    hdu.info()
    data_im1 = hdu[1].data
    # bias subtraction
    bias_subtracted_im1 = bias_subtract(hdu[1])

# first object
# Centroid detection:


# arbitrarily chosen object, section manually entered
# ymin = 1545
# ymax = 1595
# xmin = 1745
# xmax = 1795

ymin = 460
ymax = 500
xmin = 1490
xmax = 1540
object1_data = bias_subtracted_im1[ymin:ymax, xmin:xmax]

# Background subtract the object
object1_data, mask, background_dev = background_subtract(object1_data)

# show an image of the aperture
from astropy.visualization import SqrtStretch

from astropy.visualization.mpl_normalize import ImageNormalize

norm = ImageNormalize(stretch=SqrtStretch())

# plt.figure()
f1, axisarg = plt.subplots(2, 1)

axisarg[0].imshow(object1_data, norm=norm, origin='lower', cmap='viridis')
axisarg[1].imshow(mask, origin='lower', cmap='viridis')

# plt.show()


def moffat_fit(indata):
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
    offset_guess = 0

    guess = [flux_guess, x_guess, y_guess, beta_guess, a_guess, b_guess, offset_guess]

    # generate parameters for fit
    fit_result, fit_cov = curve_fit(flat_elliptical_Moffat, (x, y), indata.ravel(), p0=guess, method='lm')

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
    offset = fit_result[6]

    expected = flat_elliptical_Moffat(m_input, flux, x0, y0, beta, a, b, offset)

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
m_fit, m_cov = moffat_fit(object1_data)


print('Resultant parameters')
print('Flux: ' + str(m_fit[0]))
print('Center (x, y): '+str(m_fit[1]) + ', ' + str(m_fit[2]))
print('beta: ' + str(m_fit[3]))
print('x-axis eccentricity: ' + str(m_fit[4]))
print('y-axis eccentricity: ' + str(m_fit[5]))
print('background: ' + str(m_fit[6]))

error = np.sqrt(np.diag(m_cov))
print('Relative Error on parameters')
print(error/m_fit)

error = np.sqrt(np.diag(m_cov))
print('Error on parameters')
print(error)

# print('Covariance matrix, if that is interesting')
# print(m_cov)

print('peak count')
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
m_offset = m_fit[6]
result = elliptical_Moffat(m_input, m_flux, m_x0, m_y0, m_beta, m_a, m_b, m_offset)

# difference between the result and the observed data
result_difference = object1_data - result
# plot the result, compared to the original object
f2, axisarg = plt.subplots(2, 1)
axisarg[0].imshow(object1_data, norm=norm, origin='lower', cmap='viridis')
axisarg[1].imshow(result_difference, norm=norm, origin='lower', cmap='viridis')


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
