"""This file is for testing fit models, by generating fake data

Expect this file to be somewhat messy. Also, expect everything in this file to be unexpectedly deleted"""
# needed modules
import numpy as np
import matplotlib.pyplot as plt

# needed functions
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize


def Moffat_sum(indata, flux1, flux2, alpha1, alpha2, beta1, beta2, x0, y0, offset):
    x, y = indata
    normalize1 = (beta1-1)/(np.pi*alpha1**2)
    normalize2 = (beta2-1)/(np.pi*alpha2**2)

    moffat1 = flux1*normalize1*(1 + ((x - x0)**2 + (y - y0)**2) / (alpha1**2))**(-beta1)
    moffat2 = flux2*normalize2*(1 + ((x - x0)**2 + (y - y0)**2) / (alpha2**2))**(-beta2)
    moffat_fun = offset + moffat1 + moffat2

    return moffat_fun


def moffat_fit(indata):
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

    # generate a best guess
    y_guess = indata.shape[0] / 2
    x_guess = indata.shape[1] / 2
    amp1_guess = 50000
    amp2_guess = 50000
    beta1_guess = 1.5
    beta2_guess = 2
    alpha1_guess = 6
    alpha2_guess = 6
    offset_guess = 0
    guess = [amp1_guess, amp2_guess, alpha1_guess, alpha2_guess, beta1_guess, beta2_guess, x_guess, y_guess,
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
    upper_bounds = [np.inf, np.inf, np.inf, np.inf, 20, 20, indata.shape[1], indata.shape[0],
                    np.inf]

    bounds = (lower_bounds, upper_bounds)  # bounds set as pair of array-like tuples
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




# generate the fake object
y = np.arange(50)
x = np.arange(50)
x, y = np.meshgrid(x, y)
m_input = (x, y)
x0 = 22
y0 = 25
flux1 = 30000
flux2 = 70000
alpha1 = 6
beta1 = 1.5
alpha2 = 6
beta2 = 9
background = 0

fake_object = Moffat_sum(m_input, flux1, flux2, alpha1, alpha2, beta1, beta2, x0, y0, background)


# make a fit
m_fit, m_cov = moffat_fit(fake_object)

print('Resultant parameters')

print('flux1: ' + str(m_fit[0]))
print('flux2: ' + str(m_fit[1]))
print('alpha1: ' + str(m_fit[2]))
print('alpha2: ' + str(m_fit[3]))
print('beta1: ' + str(m_fit[4]))
print('beta2: ' + str(m_fit[5]))
print('x0: ' + str(m_fit[6]))
print('y0: ' + str(m_fit[7]))
print('background: ' + str(m_fit[8]))

# print the errors
error = np.sqrt(np.diag(m_cov))
print('Relative Error on parameters')
print(str(error/m_fit))


# generate a plot of fit result
rflux1 = m_fit[0]
rflux2 = m_fit[1]
ralpha1 = m_fit[2]
ralpha2 = m_fit[3]
rbeta1 = m_fit[4]
rbeta2 = m_fit[5]
rx0 = m_fit[6]
ry0 = m_fit[7]
rbackground = m_fit[8]

result = Moffat_sum(m_input, rflux1, rflux2, ralpha1, ralpha2, rbeta1, rbeta2, rx0, ry0, rbackground)

result_difference = fake_object - result

# show the generated object and the difference from the fit
norm = ImageNormalize(stretch=SqrtStretch())

f1, axisarg = plt.subplots(2, 1)
axisarg[0].imshow(fake_object, norm=norm, origin='lower', cmap='viridis')
axisarg[1].imshow(result_difference, norm=norm, origin='lower', cmap='viridis')

plt.show()
