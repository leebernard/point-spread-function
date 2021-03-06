

"""This file is for testing fit models, by generating fake data

Expect this file to be somewhat messy. Also, expect everything in this file to be unexpectedly deleted or changed"""
# needed modules
import numpy as np
import matplotlib.pyplot as plt

# needed functions
from astropy.visualization import SqrtStretch
# from astropy.visualization import HistEqStretch
# from astropy.visualization import LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize



def elliptical_moffat_sum(indata, flux1, flux2, a, b, beta1, beta2, x0, y0, theta):
    """Model of PSF using a single Moffat distribution, with elliptical parameters.

    Includes a parameter for axial alignment.

    """
    x_in, y_in = indata

    # moffat_fun = offset + flux * normalize * (1 + ((x - x0)**2/a**2 + (y - y0)**2/b**2))**(-beta)
    A = np.cos(theta)**2/a**2 + np.sin(theta)**2/b**2
    B = 2*np.cos(theta)*np.sin(theta)*(1/a**2 - 1/b**2)
    C = np.sin(theta)**2/a**2 + np.cos(theta)**2/b**2

    def moffat1(x, y): return (1 + A*(x - x0)**2 + B*(x - x0)*(y - y0) + C*(y - y0)**2)**(-beta1)

    def moffat2(x, y): return (1 + A*(x - x0)**2 + B*(x - x0)*(y - y0) + C*(y - y0)**2)**(-beta2)


    # numerical normalization
    # scale steps according to the size of the array.
    # produces step size of 1/10 of a pixel

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
    normalize1 = np.sum(moffat1(x_step, y_step))*delta_x*delta_y
    normalize2 = np.sum(moffat2(x_step, y_step))*delta_x*delta_y

    # forget that, just integrate it
    # normalize, norm_err = dblquad(moffat_fun, -np.inf, np.inf, lambda x: -np.inf, lambda x: np.inf)

    output = flux1*moffat1(x_in, y_in)/normalize1 + flux2*moffat2(x_in, y_in)/normalize2
    output1 = flux1*moffat1(x_in, y_in)/normalize1
    output2 = flux2*moffat2(x_in, y_in)/normalize2

    return output, output1, output2


def flat_elliptical_moffat_sum(indata, flux1, flux2, a, b, beta1, beta2, x0, y0, theta):
    """Model of PSF using a single Moffat distribution, with elliptical parameters.

    Includes a parameter for axial alignment.

    """
    x_in, y_in = indata

    A = np.cos(theta)**2/a**2 + np.sin(theta)**2/b**2
    B = 2*np.cos(theta)*np.sin(theta)*(1/a**2 - 1/b**2)
    C = np.sin(theta)**2/a**2 + np.cos(theta)**2/b**2

    def moffat1(x, y): return (1 + A*(x - x0)**2 + B*(x - x0)*(y - y0) + C*(y - y0)**2)**(-beta1)

    def moffat2(x, y): return (1 + A*(x - x0)**2 + B*(x - x0)*(y - y0) + C*(y - y0)**2)**(-beta2)


    # numerical normalization
    # scale steps according to the size of the array.
    # produces step size of 1/10 of a pixel

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
    normalize1 = np.sum(moffat1(x_step, y_step))*delta_x*delta_y
    normalize2 = np.sum(moffat2(x_step, y_step))*delta_x*delta_y

    # forget that, just integrate it
    # normalize, norm_err = dblquad(moffat_fun, -np.inf, np.inf, lambda x: -np.inf, lambda x: np.inf)

    output = flux1*moffat1(x_in, y_in)/normalize1 + flux2*moffat2(x_in, y_in)/normalize2

    return output.ravel()


def moffat_fit(indata, guess=None, bounds=None, error=None):
    """wrapper for the moffat fit procedure.

    This fit is rather complicated, so it has been wrapped into a function for convience
    """

    # fit data to moffat
    from scipy.optimize import curve_fit

    # indexes of the aperture, remembering that python indexes vert, horz
    y = np.arange(indata.shape[0])
    x = np.arange(indata.shape[1])
    x, y = np.meshgrid(x, y)


    # generate parameters for fit
    fit, cov = curve_fit(flat_elliptical_moffat_sum, (x, y), indata.ravel(), p0=guess, bounds=bounds,
                         sigma=error.ravel(), absolute_sigma=True)

    """Chi squared calculations
    """
    observed = indata.ravel()

    m_input = (x, y)
    flux1 = fit[0]
    flux2 = fit[1]
    a = fit[2]
    b = fit[3]
    beta1 = fit[4]
    beta2 = fit[5]
    x0 = fit[6]
    y0 = fit[7]
    theta = fit[8]


    expected = flat_elliptical_moffat_sum(m_input, flux1, flux2, a, b, beta1, beta2, x0, y0, theta)

    # calculated raw chi squared, including background noise
    chisq = sum(np.divide((observed - expected) ** 2, (expected + 40)))

    # degrees of freedom, 5 parameters
    degrees_of_freedom = observed.size - 9

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
flux1 = 800000
flux2 = 200000
a = 6
b = 5
beta1 = 9
beta2 = 2.9
theta = 0.7

fake_object, fake_part1, fake_part2 = elliptical_moffat_sum(m_input, flux1, flux2, a, b, beta1, beta2, x0, y0, theta)

# spike the object with some noise
background_dev = 40
noise = np.random.normal(0,background_dev,fake_object.shape)
fake_object = fake_object + noise

# make a fit #######
error = np.sqrt(fake_object + background_dev**2)
# generate a best guess
y_guess = fake_object.shape[0] / 2
x_guess = fake_object.shape[1] / 2
flux1_guess = np.sum(fake_object)*.8
flux2_guess = np.sum(fake_object)*.2
beta1_guess = 7
beta2_guess = 2
a_guess = 4
b_guess = 4
theta_guess = 0


guess = [flux1_guess, flux2_guess, a_guess, b_guess, beta1_guess, beta2_guess, x_guess, y_guess, theta_guess]

# create bounds for the fit, in an attempt to keep it from blowing up
"""
flux1_bound = [0, np.inf]
flux2_bound = [0, np.inf]
a_bound = [0.1, 20]
b_bound = [0.1, 20
beta1_bound = [1, 20]
beta2_bound = [1, 20]
x_bound = [0, object1_data.shape[1]]
y_bound = [0, object1_data.shape[0]]
theta_bound = [0, np.pi/2]
"""
# format the bounds
lower_bounds = [0, 0, 0.1, .1, 1, 1, 0, 0, 0]
upper_bounds = [np.inf, np.inf, 20, 20, 20, 20, fake_object.shape[1], fake_object.shape[0], np.pi/2]

bounds = (lower_bounds, upper_bounds)  # bounds set as pair of array-like tuples

m_fit, m_cov = moffat_fit(fake_object, guess=guess, bounds=bounds, error=error)

# calculate errors
fit_error = np.sqrt(np.diag(m_cov))
print('Resultant parameters')

print(f'flux1: {m_fit[0]: .2f}±{fit_error[0]:.2f} (Actual: {flux1})')
print(f'flux2: {m_fit[1]: .2f}±{fit_error[1]:.2f} (Actual: {flux2})')
print(f'a: {m_fit[2]: .2f}±{fit_error[2]:.2f} (Actual: {a})')
print(f'b: {m_fit[3]: .2f}±{fit_error[3]:.2f} (Actual: {b})')
print(f'beta1: {m_fit[4]: .2f}±{fit_error[4]:.2f} (Actual: {beta1})')
print(f'beta2: {m_fit[5]: .2f}±{fit_error[5]:.2f} (Actual: {beta2})')
print(f'x0: {m_fit[6]: .2f}±{fit_error[6]:.2f} (Actual: {x0})')
print(f'y0: {m_fit[7]: .2f}±{fit_error[7]:.2f} (Actual: {y0})')
print(f'theta: {m_fit[8]: .2f}±{fit_error[8]:.2f} (Actual: {theta})')
# f', starting guess: {flux1_guess})')
# f', starting guess: {flux2_guess})')
# f', starting guess: {alpha_guess})')
# f', starting guess: {alpha2_guess})')
# f', starting guess: {beta1_guess})')
# f', starting guess: {beta2_guess})')
# f', starting guess: {x_guess})')
# f', starting guess: {y_guess})')


# print the errors
#
# print('Relative Error on parameters')
# print(str(fit_error/m_fit))


# generate a plot of fit results
rflux1 = m_fit[0]
rflux2 = m_fit[1]
ra = m_fit[2]
rb = m_fit[3]
rbeta1 = m_fit[4]
rbeta2 = m_fit[5]
rx0 = m_fit[6]
ry0 = m_fit[7]
rtheta = m_fit[8]

result, result_part1, result_part2 = elliptical_moffat_sum(m_input, rflux1, rflux2, ra, rb, rbeta1, rbeta2, rx0, ry0, rtheta)

result_difference = fake_object - result
difference_part1 = -(fake_part1 - result_part1)
if (np.amax(difference_part1) + np.amin(difference_part1)) < 0:
    difference_part1 = difference_part1*-1

difference_part2 = fake_part2 - result_part2
if (np.amax(difference_part2) + np.amin(difference_part2)) < 0:
    difference_part2 = -difference_part2


# bias the data to remove negatives

# show the generated object and the difference from the fit
norm = ImageNormalize(stretch=SqrtStretch())


f1, axisarg = plt.subplots(3, 3, figsize=(12, 12))
# show the total object
fake_object_plt = axisarg[0][0].imshow(fake_object, norm=norm, origin='lower', cmap='viridis')
axisarg[0][0].set_title('Fake Object, sum of two Moffats')
# f1.colorbar(fake_object_plt, ax=axisarg[0][0])

result_plt = axisarg[0][1].imshow(result, norm=norm, origin='lower', cmap='viridis')
axisarg[0][1].set_title('Result of fit')
# f1.colorbar(result_plt, ax=axisarg[0][1])

difference_plt = axisarg[0][2].imshow(result_difference, norm=norm, origin='lower', cmap='viridis')
axisarg[0][2].set_title('Residuals of fake object and fit result')
f1.colorbar(difference_plt, ax=axisarg[0][2])

# show the first moffat
part1_plt = axisarg[1][0].imshow(fake_part1, norm=norm, origin='lower', cmap='viridis')
axisarg[1][0].set_title('Part 1 of Moffat distro')
# f1.colorbar(part1_plt, ax=axisarg[1][0])

result1_plt = axisarg[1][1].imshow(result_part1, norm=norm, origin='lower', cmap='viridis')
axisarg[1][1].set_title('Result of fit, part 1')
# f1.colorbar(result1_plt, ax=axisarg[1][1])

difference1_plt = axisarg[1][2].imshow(difference_part1, norm=norm, origin='lower', cmap='viridis')
axisarg[1][2].set_title('Residuals of part 1')
f1.colorbar(difference1_plt, ax=axisarg[1][2])

# show the second moffat
part2_plt = axisarg[2][0].imshow(fake_part2, norm=norm, origin='lower', cmap='viridis')
axisarg[2][0].set_title('Part 2 of Moffat distro')

result2_plt = axisarg[2][1].imshow(result_part2, norm=norm, origin='lower', cmap='viridis')
axisarg[2][1].set_title('Result of fit, part 2')

difference2_plt = axisarg[2][2].imshow(difference_part2, norm=norm, origin='lower', cmap='viridis')
axisarg[2][2].set_title('Residuals of part 2')
f1.colorbar(difference2_plt, ax=axisarg[2][2])

plt.show()
