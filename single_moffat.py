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
object1_data_pass, mask = background_subtract(object1_data)

# centroid techniques: need to learn the difference3s
# print('Centroids:')
# x1, y1 = centroid_com(object1_data)
# print((x1, y1))
# x2, y2 = centroid_1dg(object1_data)
# print((x2, y2))
# x3, y3 = centroid_2dg(object1_data)
# print((x3, y3))

# show an image of the aperture
from astropy.visualization import SqrtStretch

from astropy.visualization.mpl_normalize import ImageNormalize
norm = ImageNormalize(stretch=SqrtStretch())

# plt.figure()
f1, axisarg = plt.subplots(2,1)

axisarg[0].imshow(object1_data, norm=norm, origin='lower', cmap='viridis')
axisarg[1].imshow(mask, origin='lower', cmap='viridis')
# plt.show()
#
#


"""Centroid dectection by curve fitting

this attempt uses a bivariate normal distorbution as a model for the object

define gaussian function, assuming no correlation between x and y
indata is a pair of arrays, each array corresponding to the x indice or y indice, in the form (x, y)
amplitude is the maximum amplitude of the function, minus background
x0, y0 are the center coord. of the function
sigma_x, sigma_y are the widths of the function
offset is the background
the output is flattened, in order to package it for curve_fit
"""
"""
def Gaussian_2d(indata, amplitude, x0, y0, sigma_x, sigma_y, offset):
    import numpy as np
    x, y = indata
    normalize = 1 / (sigma_x * sigma_y * 2 * np.pi)

    gaussian_fun = offset + amplitude*normalize*np.exp(-(x-x0)**2/(2*sigma_x**2) - (y-y0)**2/(2*sigma_y**2))

    return gaussian_fun.ravel()
"""


def Moffat(indata, flux, x0, y0, alpha, beta, offset):
    """define sum of two Moffat function for curve fitting
    """
    x, y = indata
    normalize = (beta-1)/(np.pi*alpha**2)

    moffat_fun = offset + flux*normalize*(1 + ((x-x0)**2 + (y-y0)**2)/(alpha**2))**(-beta)

    return moffat_fun.ravel()


# fit data to gaussian
# instead fit data to moffat
from scipy.optimize import curve_fit

# indexes of the aperture, remembering that python indexes vert, horz
y = np.arange(object1_data.shape[0])
x = np.arange(object1_data.shape[1])
x, y = np.meshgrid(x, y)

# generate a best guess
flux_guess = np.amax(object1_data)
y_guess = object1_data.shape[0]/2
x_guess = object1_data.shape[1]/2
alpha_guess = 4
beta_guess = 2
offset_guess = 0

guess = [flux_guess, x_guess, y_guess, alpha_guess, beta_guess, offset_guess]

# generate parameters for fit
m_fit, m_cov = curve_fit(Moffat, (x, y), object1_data.ravel(), p0=guess)
print('Resultant parameters')
print(m_fit)

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
x_width = m_fit[3] # alpha is the width
y_width = m_fit[3]

plt.figure()
plt.imshow(object1_data, norm=norm, origin='lower', cmap='viridis')
plt.errorbar(x_center, y_center, xerr=x_width, yerr=y_width, ecolor='red')


"""Chi squared calculations
"""
observed = object1_data.ravel()

m_input = (x, y)
flux = m_fit[0]
x0 = m_fit[1]
y0 = m_fit[2]
alpha = m_fit[3]
beta = m_fit[4]
offset = m_fit[5]

expected = Moffat(m_input, flux, x0, y0, alpha, beta, offset)

# calculated raw chi squared
chisq = sum(np.divide((observed - expected)**2, expected))

# degrees of freedom, 5 parameters
degrees_of_freedom = observed.size - 6

# normalized chi squared
chisq_norm = chisq/degrees_of_freedom

print('normalized chi squared:')
print(chisq_norm)

# histogram
# plt.figure()
# histogram = plt.hist(object1_data.flatten(),bins=2000, range=[-500, 30000])
#
# plt.figure()
# plt.hist(object1_data.flatten(),bins=2000, range=[-500, 30000])
plt.show() # show all figures
