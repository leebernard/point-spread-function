"""This file is for testing fit models, by generating fake data

Expect this file to be somewhat messy. Also, expect everything in this file to be unexpectedly deleted"""
# needed modules
import numpy as np
import matplotlib.pyplot as plt

# needed functions
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize


# functions to be tested
def Moffat(indata, flux, x0, y0, alpha, beta, offset):
    """Model of PSF using a single Moffat distribution
    """
    x, y = indata
    normalize = (beta - 1) / (np.pi * alpha ** 2)

    moffat_fun = offset + flux*normalize*(1 + ((x - x0)**2 + (y - y0)**2) / (alpha**2))**(-beta)

    return moffat_fun


def flat_Moffat(indata, flux, x0, y0, alpha, beta, offset):
    """Model of PSF using a single Moffat distribution

    This function flattens the output, for curve fitting
    """
    x, y = indata
    normalize = (beta-1)/(np.pi*alpha**2)

    moffat_fun = offset + flux*normalize*(1 + ((x-x0)**2 + (y-y0)**2)/(alpha**2))**(-beta)

    return moffat_fun.ravel()



def elliptical_Moffat(indata, flux, x0, y0, beta, a, b, offset):
    """Model of PSF using a single Moffat distribution, with elliptical parameters.

    This version does not account for axial alignment.

    This function flattens the output, for curve fitting
    """
    x, y = indata
    normalize = 1  #(beta - 1) / ((a*b) * np.pi)

    moffat_fun = offset + flux * normalize * (1 + ((x - x0)**2/a**2 + (y - y0)**2/b**2))**(-beta)

    return moffat_fun


def flat_elliptical_Moffat(indata, flux, x0, y0, beta, a, b, offset):
    """Model of PSF using a single Moffat distribution, with elliptical parameters.

    This version does not account for axial alignment

    This function flattens the output, for curve fitting

    """
    x, y = indata
    normalize = 1  #(beta - 1) / ((a*b) * np.pi)

    moffat_fun = offset + flux * normalize * (1 + ((x - x0)**2/a**2 + (y - y0)**2/b**2))**(-beta)

    return moffat_fun.ravel()


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
    y_guess = indata.shape[0]/2
    x_guess = indata.shape[1]/2
    beta_guess = 2
    a_guess = 2
    b_guess = 2
    offset_guess = 0

    guess = [flux_guess, x_guess, y_guess, beta_guess, a_guess, b_guess, offset_guess]

    # create bounds for the fit, in an attempt to keep it from blowing up
    """
    flux_bound = [0, np.inf]
    x_bound = [0, object1_data.shape[1]]
    y_bound = [0, object1_data.shape[0]]
    beta_bound = [1, 20]]
    a_bound = [0.1, np.inf]
    b_bound = [0.1, np.inf]
    offset_bound = [-np.inf, np.inf]
    """
    # format the bounds
    lower_bounds = [0, 0, 0, 1, 0.1, 0.1, -np.inf]
    upper_bounds = [np.inf, indata.shape[1], indata.shape[0], 20, np.inf, np.inf,
                    np.inf]
    bounds = (lower_bounds, upper_bounds)  # bounds set as pair of array-like tuples

    # generate parameters for fit
    fit_result, fit_cov = curve_fit(flat_elliptical_Moffat, (x, y), indata.ravel(), p0=guess, bounds=bounds)
    # fit_result, fit_cov = curve_fit(flat_Moffat, (x, y), indata.ravel(), p0=guess)

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
    # expected = flat_Moffat(m_input, flux, x0, y0, alpha, beta, offset)
    # calculated raw chi squared
    chisq = sum(np.divide((observed - expected) ** 2, expected))

    # degrees of freedom, 5 parameters
    degrees_of_freedom = observed.size - 6

    # normalized chi squared
    chisq_norm = chisq / degrees_of_freedom

    print('normalized chi squared:')
    print(chisq_norm)
    return fit_result, fit_cov


# generate the fake object
y = np.arange(50)
x = np.arange(50)
x, y = np.meshgrid(x, y)
m_input = (x, y)
flux = 1000000  # 1 million
x0 = 26
y0 = 22
beta = 5
a = 2
b = 3.6
offset = 0
fake_object = elliptical_Moffat(m_input, flux, x0, y0, beta, a, b, offset)

# spike the object with some noise
noise = np.random.normal(0,10,fake_object.shape)
# fake_object = fake_object + noise



# fit the fake data
m_fit, m_cov = moffat_fit(fake_object)

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

# generate the data from the result fit
result = elliptical_Moffat(m_input, m_fit[0], m_fit[1], m_fit[2], m_fit[3], m_fit[4], m_fit[5], m_fit[6])

# difference from the fake object
result_difference = fake_object-result

# show the generated object and the difference from the fit
norm = ImageNormalize(stretch=SqrtStretch())

f1, axisarg = plt.subplots(2, 1)
axisarg[0].imshow(fake_object, norm=norm, origin='lower', cmap='viridis')
axisarg[1].imshow(result, norm=norm, origin='lower', cmap='viridis')


plt.show()