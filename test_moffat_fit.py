

"""This file is for testing fit models, by generating fake data

Expect this file to be somewhat messy. Also, expect everything in this file to be unexpectedly deleted"""
# needed modules
import numpy as np
import matplotlib.pyplot as plt

# needed functions
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize

from scipy.integrate import dblquad


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


def elliptical_Moffat(indata, flux, x0, y0, alpha, beta, roh, theta, offset):
    """Model of PSF using a single Moffat distribution, with elliptical parameters.

    Includes a parameter for axial alignment.

    """
    x_in, y_in = indata

    # moffat_fun = offset + flux * normalize * (1 + ((x - x0)**2/a**2 + (y - y0)**2/b**2))**(-beta)
    A = np.cos(theta)**2/(roh*alpha)**2 + np.sin(theta)**2/alpha**2
    B = 2*np.cos(theta)*np.sin(theta)*(1/(roh*alpha)**2 - 1/alpha**2)
    C = np.sin(theta)**2/(roh*alpha)**2 + np.cos(theta)**2/alpha**2

    def moffat_fun(x, y): return (1 + A*(x - x0)**2 + B*(x - x0)*(y - y0) + C*(y - y0)**2)**(-beta)

    # numerical normalization
    # scale steps according to the size of the array.
    # produces step size of 1/10 of a pixel

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

    #forget that, just integrate it
    # normalize, norm_err = dblquad(moffat_fun, -np.inf, np.inf, lambda x: -np.inf, lambda x: np.inf)

    output = offset + flux*moffat_fun(x_in, y_in)/normalize

    return output


def flat_elliptical_Moffat(indata, flux, x0, y0, alpha, beta, roh, theta, offset):
    """Model of PSF using a single Moffat distribution, with elliptical parameters.

    Includes a parameter for  axial alignment. This function flattens the output, for curve fitting.

    """
    x_in, y_in = indata

    # moffat_fun = offset + flux * normalize * (1 + ((x - x0)**2/a**2 + (y - y0)**2/b**2))**(-beta)
    A = np.cos(theta)**2/(roh*alpha)**2 + np.sin(theta)**2/alpha**2
    B = 2*np.cos(theta)*np.sin(theta)*(1/(roh*alpha)**2 - 1/alpha**2)
    C = np.sin(theta)**2/(roh*alpha)**2 + np.cos(theta)**2/alpha**2

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

    #forget that, just integrate it
    # normalize, norm_err = dblquad(moffat_fun, -np.inf, np.inf, lambda x: -np.inf, lambda x: np.inf)

    output = offset + flux*moffat_fun(x_in, y_in)/normalize


    return output.ravel()


def moffat_fit(indata):
    """wrapper for the moffat fit procedure.

    This fit is rather complicated, so it has been wrapped into a function for convenience
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
    alpha_guess = 4
    beta_guess = 2
    roh_guess = 1
    theta_guess = 0
    offset_guess = 0

    guess = [flux_guess, x_guess, y_guess, alpha_guess, beta_guess, roh_guess, theta_guess, offset_guess]

    # create bounds for the fit, in an attempt to keep it from blowing up
    """
    flux_bound = [0, np.inf]
    x_bound = [0, object1_data.shape[1]]
    y_bound = [0, object1_data.shape[0]]
    alpha_bound = [0.1, np.inf]]
    beta_bound = [1.1, 20]
    roh_bound = [0, np.inf]
    theta_bound = 0, np.pi/2]
    offset_bound = [-np.inf, np.inf]
    """
    # format the bounds
    lower_bounds = [0, 0, 0, 0.1, 1.1, 0, 0, -np.inf]
    upper_bounds = [np.inf, indata.shape[1], indata.shape[0], np.inf, 20, np.inf, np.pi/2, np.inf]
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
    alpha = fit_result[3]
    beta = fit_result[4]
    roh = fit_result[5]
    theta = fit_result[6]
    offset = fit_result[7]

    expected = flat_elliptical_Moffat(m_input, flux, x0, y0, alpha, beta, roh, theta, offset)
    # expected = flat_Moffat(m_input, flux, x0, y0, alpha, beta, offset)
    # calculated raw chi squared
    chisq = sum(np.divide((observed - expected) ** 2, expected + 40))

    # degrees of freedom, 5 parameters
    degrees_of_freedom = observed.size - 6

    # normalized chi squared
    chisq_norm = chisq / degrees_of_freedom

    print('normalized chi squared:')
    print(chisq_norm)
    return fit_result, fit_cov



# generate the fake object
y = np.arange(40)
x = np.arange(50)
x, y = np.meshgrid(x, y)
m_input = (x, y)
flux = 1000000  # 1 million
x0 = 26
y0 = 22
alpha = 6
beta = 5
roh = 1.3
theta = .707
offset = 0
fake_object = elliptical_Moffat(m_input, flux, x0, y0, alpha, beta, roh, theta, offset)

# spike the object with some noise
noise = np.random.normal(0,40,fake_object.shape)
fake_object = fake_object + noise



# fit the fake data
m_fit, m_cov = moffat_fit(fake_object)


error = np.sqrt(np.diag(m_cov))


print('Resultant parameters')
print(f'flux: {m_fit[0]: .2f}±{error[0]:.2f} (Actual: {flux})')
print(f'x0: {m_fit[1]: .2f}±{error[1]:.2f} (Actual: {x0})')
print(f'y0: {m_fit[2]: .2f}±{error[2]:.2f} (Actual: {y0})')
print(f'alpha1: {m_fit[3]: .2f}±{error[3]:.2f} (Actual: {alpha})')
print(f'beta: {m_fit[4]: .2f}±{error[4]:.2f} (Actual: {beta})')
print(f'eccentricity: {m_fit[5]: .2f}±{error[5]:.2f} (Actual: {roh})')
print(f'angle of eccentricity:  {m_fit[6]: .2f}±{error[6]:.2f} (Actual: {theta})')
print(f'background: {m_fit[7]: .2f}±{error[7]:.2f} (Actual: {offset})')

# print('Relative Error on parameters')
# print(error/m_fit)

# generate the data from the result fit
result = elliptical_Moffat(m_input, m_fit[0], m_fit[1], m_fit[2], m_fit[3], m_fit[4], m_fit[5], m_fit[6], m_fit[7])

# difference from the fake object
result_difference = fake_object-result

# show the generated object and the difference from the fit
norm = ImageNormalize(stretch=SqrtStretch())

f1, axisarg = plt.subplots(3, 1, figsize=(10,10))
fake_object_plt = axisarg[0].imshow(fake_object, norm=norm, origin='lower', cmap='viridis')
axisarg[0].set_title('Fake Object, single elliptical Moffat')
f1.colorbar(fake_object_plt, ax=axisarg[0])

fit_plt = axisarg[1].imshow(result, norm=norm, origin='lower', cmap='viridis')
axisarg[1].set_title('Resultant fit of object')
f1.colorbar(fit_plt, ax=axisarg[1])

residual_plt = axisarg[2].imshow(result_difference, norm=norm, origin='lower', cmap='viridis')
axisarg[2].set_title('Residuals')
f1.colorbar(residual_plt, ax=axisarg[2])


"""Method for showing the plots and the bins on the same figure

plt.figure()
#first plot, bars to represent pixels
x_slice = fake_object[23][:]
bar_domain = np.arange(x_slice.shape[0])
plt.bar(bar_domain, x_slice)
# second plot, to show the fit curve
plot_domain = np.arange(0, x_slice.shape[0], .01)
y_values = np.ones(plot_domain.size) * 23  # multiplied by y value of slice
plot_input = (plot_domain, y_values)
plot_range = elliptical_Moffat(plot_input, m_fit[0], m_fit[1], m_fit[2], m_fit[3], m_fit[4], m_fit[5], m_fit[6])
plt.plot(plot_domain, plot_range)
"""
plt.show()