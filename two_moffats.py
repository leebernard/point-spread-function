
"""This file is for testing fit models on real sky data
"""
# needed modules
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# needed functions
from astropy.visualization import SqrtStretch
# from astropy.visualization import HistEqStretch
# from astropy.visualization import LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize
# import needed functions from the toolbox
from ccd_tools import bias_subtract, background_subtract


def Moffat_sum(indata, flux1, flux2, alpha1, alpha2, beta1, beta2, x0, y0, offset):
    x, y = indata
    normalize1 = (beta1-1)/(np.pi*alpha1**2)
    normalize2 = (beta2-1)/(np.pi*alpha2**2)

    moffat1 = flux1*normalize1*(1 + ((x - x0)**2 + (y - y0)**2) / (alpha1**2))**(-beta1)
    moffat2 = flux2*normalize2*(1 + ((x - x0)**2 + (y - y0)**2) / (alpha2**2))**(-beta2)
    moffat_fun = offset + moffat1 + moffat2

    return moffat_fun, moffat1, moffat2


def moffat_fit(indata, guess=None, bounds=None):
    """wrapper for the moffat fit procedure.

    This fit is rather complicated, so it has been wrapped into a function for convience
    """

    def flat_Moffat_sum(indata, flux1, flux2, alpha1, alpha2, beta1, beta2, x0, y0, offset):
        x, y = indata
        normalize1 = (beta1 - 1) / (np.pi * alpha1 ** 2)
        normalize2 = (beta2 - 1) / (np.pi * alpha2 ** 2)

        moffat1 = flux1 * normalize1 * (1 + ((x - x0) ** 2 + (y - y0) ** 2) / (alpha1 ** 2)) ** (-beta1)
        moffat2 = flux2 * normalize2 * (1 + ((x - x0) ** 2 + (y - y0) ** 2) / (alpha2 ** 2)) ** (-beta2)
        moffat_fun = offset + moffat1 + moffat2

        return moffat_fun.ravel()

    # fit data to moffat
    from scipy.optimize import curve_fit

    # indexes of the aperture, remembering that python indexes vert, horz
    y = np.arange(indata.shape[0])
    x = np.arange(indata.shape[1])
    x, y = np.meshgrid(x, y)




    # generate parameters for fit
    fit, cov = curve_fit(flat_Moffat_sum, (x, y), indata.ravel(), bounds=bounds, p0=guess)


    """Chi squared calculations
    """
    observed = indata.ravel()

    m_input = (x, y)
    flux1 = fit[0]
    flux2 = fit[1]
    alpha1 = fit[2]
    alpha2 = fit[3]
    beta1 = fit[4]
    beta2 = fit[5]
    x0 = fit[6]
    y0 = fit[7]
    background = fit[8]

    expected = flat_Moffat_sum(m_input, flux1, flux2, alpha1, alpha2, beta1, beta2, x0, y0, background)

    # calculated raw chi squared
    chisq = sum(np.divide((expected - observed) ** 2, (observed)))

    # degrees of freedom, 5 parameters
    degrees_of_freedom = observed.size - 6

    # normalized chi squared
    chisq_norm = chisq / degrees_of_freedom

    print('normalized chi squared:')
    print(chisq_norm)
    return fit, cov


# generate a real object
# open fits file, best practice
# file_name = '/home/lee/Documents/k4m_160319_101212_ori.fits.fz'
# found a better file
file_name = '/home/lee/Documents/k4m_160531_050920_ori.fits.fz'
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
#
# ymin = 460
# ymax = 500
# xmin = 1490
# xmax = 1540

# this object has a low eccentricity
# ymin = 1306
# ymax = 1356
# xmin = 1636
# xmax = 1706

# trying with a different object
ymin = 1777
ymax = 1822
xmin = 252
xmax = 302
real_boy = bias_subtracted_im1[ymin:ymax, xmin:xmax]

# Background subtract the object
real_boy, mask, background_dev = background_subtract(real_boy)


# make a fit #######

# generate a best guess
y_guess = real_boy.shape[0] / 2
x_guess = real_boy.shape[1] / 2
flux1_guess = np.sum(real_boy)
flux2_guess = flux1_guess/10
beta1_guess = 7
beta2_guess = 2
alpha1_guess = 2
alpha2_guess = 2
offset_guess = 0
guess = [flux1_guess, flux2_guess, alpha1_guess, alpha2_guess, beta1_guess, beta2_guess, x_guess, y_guess,
         offset_guess]

# create bounds for the fit, in an attempt to keep it from blowing up
"""
flux1_bound = [0, np.inf]
flux2_bound = [0, np.inf]
alpha1_bound = [0.1, np.inf]
alpha2_bound = [0.1, np.inf]
beta1_bound = [1, 20]
beta2_bound = [1, 20]
x_bound = [0, object1_data.shape[1]]
y_bound = [0, object1_data.shape[0]]
offset_bound = [-np.inf, np.inf]
"""
# format the bounds
lower_bounds = [0, 0, 0.1, 0.1, 1, 1, 0, 0, -np.inf]
upper_bounds = [np.inf, np.inf, 12, 12, 20, 20, real_boy.shape[1], real_boy.shape[0],
                np.inf]

bounds = (lower_bounds, upper_bounds)  # bounds set as pair of array-like tuples

m_fit, m_cov = moffat_fit(real_boy, guess=guess, bounds=bounds)

print('Resultant parameters')

print(f'flux1: {m_fit[0]: .2f} (starting guess: {flux1_guess})')
print(f'flux2: {m_fit[1]: .2f} (starting guess: {flux2_guess})')
print(f'alpha1: {m_fit[2]: .2f} (starting guess: {alpha1_guess})')
print(f'alpha2: {m_fit[3]: .2f} (starting guess: {alpha2_guess})')
print(f'beta1: {m_fit[4]: .2f} (starting guess: {beta1_guess})')
print(f'beta2: {m_fit[5]: .2f} (starting guess: {beta2_guess})')
print(f'x0: {m_fit[6]: .2f} (starting guess: {x_guess})')
print(f'y0: {m_fit[7]: .2f} (starting guess: {y_guess})')
print(f'background: {m_fit[8]: .2f} (starting guess: {offset_guess})')

# print the errors
error = np.sqrt(np.diag(m_cov))
print('Relative Error on parameters')
print(str(error/m_fit))


# generate a plot of fit results
# Calculate the resultant 2d spread
y = np.arange(real_boy.shape[0])
x = np.arange(real_boy.shape[1])
x, y = np.meshgrid(x, y)
rinput = (x, y)
rflux1 = m_fit[0]
rflux2 = m_fit[1]
ralpha1 = m_fit[2]
ralpha2 = m_fit[3]
rbeta1 = m_fit[4]
rbeta2 = m_fit[5]
rx0 = m_fit[6]
ry0 = m_fit[7]
rbackground = m_fit[8]

result, result_part1, result_part2 = Moffat_sum(rinput, rflux1, rflux2, ralpha1, ralpha2, rbeta1, rbeta2, rx0, ry0, rbackground)

norm = ImageNormalize(stretch=SqrtStretch())

# difference between the result and the observed data
result_difference = real_boy - result
# plot the result, compared to the original object
f2, axisarg = plt.subplots(2, 1)
object_im = axisarg[0].imshow(real_boy, norm=norm, origin='lower', cmap='viridis')
residual_im = axisarg[1].imshow(result_difference, norm=norm, origin='lower', cmap='viridis')
f2.colorbar(object_im, ax=axisarg[0])
f2.colorbar(residual_im, ax=axisarg[1])

plt.show()