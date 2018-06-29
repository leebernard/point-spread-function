

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


"""Centroid dectection by curve fitting, using a bivariate normal distribution as a model for the object

"""
"""
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


def Moffat_sum(indata, flux1, flux2, alpha1, alpha2, beta1, beta2, x0, y0, offset):
    x, y = indata
    normalize1 = (beta1-1)/(np.pi*alpha1**2)
    normalize2 = (beta2-1)/(np.pi*alpha2**2)

    moffat1 = flux1*normalize1*(1 + ((x - x0)**2 + (y - y0)**2) / (alpha1**2))**(-beta1)
    moffat2 = flux2*normalize2*(1 + ((x - x0)**2 + (y - y0)**2) / (alpha2**2))**(-beta2)
    moffat_fun = offset + moffat1 + moffat2

    return moffat_fun.ravel()

# fit data to gaussian
# instead fit data to moffat
from scipy.optimize import curve_fit

# indexes of the aperture, remembering that python indexes vert, horz
y = np.arange(object1_data.shape[0])
x = np.arange(object1_data.shape[1])
x, y = np.meshgrid(x, y)

# generate a best guess
y_guess = object1_data.shape[0]/2
x_guess = object1_data.shape[1]/2
amp1_guess = np.amax(object1_data)
amp2_guess = np.mean(object1_data)
beta1_guess = 1000
beta2_guess = 2
alpha1_guess = 2
alpha2_guess = 20
offset_guess = 0
guess = [amp1_guess, amp2_guess, alpha1_guess, alpha2_guess, beta1_guess, beta2_guess, x_guess, y_guess, offset_guess]

# create bounds for the fit, in an attempt to keep it from blowing up
"""
flux1_bound = [0, np.inf]
flux2_bound = [0, np.inf]
alpha1_bound = [0.1, np.inf]
alpha2_bound = [0.1, np.inf]
beta1_bound = [1, np.inf]
beta2_bound = [1, np.inf]
x_bound = [0, object1_data.shape[1]]
y_bound = [0, object1_data.shape[0]]
offset_bound = [-np.inf, np.inf]
"""
# format the bounds
lower_bounds = [0, 0, 0.1, 0.1, 1, 1, 0, 0, -np.inf]
upper_bounds = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, object1_data.shape[1], object1_data.shape[0], np.inf]

bounds = (lower_bounds, upper_bounds)  # bounds set as pair of array-like tuples

# curve fit
# m_fit, m_cov = curve_fit(Moffat, (x,y), object1_data.ravel(), p0=[amp1_guess, x_guess, y_guess, 2, 2, 2])
m_fit, m_cov = curve_fit(Moffat_sum, (x, y), object1_data.ravel(), p0=guess, bounds=bounds)
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

"""this is the center for the single moffat
x_center = m_fit[1]
y_center = m_fit[2]
x_width = m_fit[3] # alpha is the width
y_width = m_fit[3]
"""
x_center = m_fit[6]
y_center = m_fit[7]
x_width = m_fit[3] # width of narrow moffet
y_width = m_fit[2] # width of wide moffet
plt.figure()
plt.imshow(object1_data, norm=norm, origin='lower', cmap='viridis')
plt.errorbar(x_center, y_center, xerr=x_width, yerr=y_width, ecolor='red')


# print('center: ',x_center, ',', y_center)
# print('width: ', 2*x_width, 'by', 2*y_width)


"""Chi squared calculations
"""
observed = object1_data.ravel()

# define the inputs for the Moffat
m_input = (x, y)
flux1 = m_fit[0]
flux2 = m_fit[1]
alpha1 = m_fit[2]
alpha2 = m_fit[3]
beta1 = m_fit[4]
beta2 = m_fit[5]
x0 = m_fit[6]
y0 = m_fit[7]
offset = m_fit[8]
# expected = Moffat(m_input, flux, x0, y0, alpha, beta, offset) # old, for single Moffat

expected = Moffat_sum(m_input, flux1, alpha1, beta1, flux2, alpha2, beta2, x0, y0, offset)

# calculated raw chi squared
chisq = sum(np.divide((observed - expected)**2, expected))

# degrees of freedom, 5 parameters
degrees_of_freedom = observed.size - 9

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
