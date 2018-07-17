

# needed packages
import numpy as np
# import matplotlib
import matplotlib.pyplot as plt
# import re
from astropy.io import fits
import pyds9
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
# import needed functions from the toolbox
from ccd_tools import bias_subtract, background_subtract, get_regions


'''
def Moffat_sum(indata, flux1, flux2, alpha, beta1, beta2, x0, y0):
    x, y = indata
    normalize1 = (beta1-1)/(np.pi*alpha**2)
    normalize2 = (beta2-1)/(np.pi*alpha**2)

    moffat1 = flux1*normalize1*(1 + ((x - x0)**2 + (y - y0)**2)/(alpha**2))**(-beta1)
    moffat2 = flux2*normalize2*(1 + ((x - x0)**2 + (y - y0)**2)/(alpha**2))**(-beta2)
    moffat_fun = moffat1 + moffat2

    return moffat_fun, moffat1, moffat2


def flat_Moffat_sum(indata, flux1, flux2, alpha, beta1, beta2, x0, y0):
    x, y = indata
    normalize1 = (beta1-1)/(np.pi*alpha**2)
    normalize2 = (beta2-1)/(np.pi*alpha**2)

    moffat1 = flux1*normalize1*(1 + ((x - x0)**2 + (y - y0)**2)/(alpha**2))**(-beta1)
    moffat2 = flux2*normalize2*(1 + ((x - x0)**2 + (y - y0)**2)/(alpha**2))**(-beta2)
    moffat_fun = moffat1 + moffat2

    return moffat_fun.ravel()
'''

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


# check if ds9 is accesible
if not pyds9.ds9_targets():
    input('DS9 target not found. Please start/restart DS9, then press enter')

# show the ds9 target
print('ds9 target instance')
print(pyds9.ds9_targets())

# create a DS9 object
ds9 = pyds9.DS9()

# import the current fits file loaded in DS9
hdu = ds9.get_pyfits()

hdu.info()

# import a list of region definitions
selected_regions = get_regions(get_data=False)

# sort the regions according to the distance from origin
selected_regions.sort(key=lambda region: np.sqrt(region.x_coord**2 + region.y_coord**2))

# get the bias subtracted data
bias_subtracted_data = bias_subtract(hdu[0])
gain = hdu[0].header['GAIN']

# use the regions to produce apertures of thedata
# also background subtract the data
aperture_list = []  # list for holding aperture data
for region in selected_regions:
    current_aperture = bias_subtracted_data[region.ymin:region.ymax, region.xmin:region.xmax]

    # convert to electron count
    current_aperture = current_aperture*gain
    aperture_list.append(current_aperture)


"""calculate the centroid for each aperture"""

n = 0  # counting variable
# generate each curve fit
from scipy.optimize import curve_fit
for aperture in aperture_list:

    print('---------------------')
    print('Region ' + str(n+1) + ': ' + selected_regions[n].region_def)
    n += 1
    # background subtract the aperture
    aperture, mask, background_dev = background_subtract(aperture)

    # generate the associated pixel error
    aperture_err = np.sqrt(aperture + background_dev**2)

    # plot the aperture and mask used to background subtract
    norm = ImageNormalize(stretch=SqrtStretch())
    f1, axisarg = plt.subplots(3, 2, figsize=(10, 10))
    aperture_im = axisarg[0][0].imshow(aperture, norm=norm, origin='lower', cmap='viridis')
    f1.colorbar(aperture_im, ax=axisarg[0][0])
    axisarg[0][0].set_title('Object, with colorscale intensity')
    mask_im = axisarg[1][0].imshow(mask, origin='lower', cmap='viridis')
    axisarg[1][0].set_title('Mask used in background calculations')
    aperture_hist = axisarg[1][1].hist(aperture.flatten(),bins=500, range=[-100, 5000])
    axisarg[1][1].set_title('Object histogram after background subtraction')

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
    upper_bounds = [np.inf, np.inf, 20, 20, 20, 20, aperture.shape[1], aperture.shape[0], np.pi / 2]

    bounds = (lower_bounds, upper_bounds)  # bounds set as pair of array-like tuples

    # generate a best guess
    y_guess = aperture.shape[0] / 2
    x_guess = aperture.shape[1] / 2
    flux1_guess = np.sum(aperture)*0.8
    flux2_guess = np.sum(aperture)*0.2
    beta1_guess = 7
    beta2_guess = 2
    a_guess = 4
    b_guess = 4
    theta_guess = 0

    guess = [flux1_guess, flux2_guess, a_guess, b_guess, beta1_guess, beta2_guess, x_guess, y_guess, theta_guess]

    # indexes of the apature, remembering that python indexes vert, horz
    y = np.arange(aperture.shape[0])
    x = np.arange(aperture.shape[1])
    x, y = np.meshgrid(x, y)


    # curve fit
    try:
        m_fit, m_cov = curve_fit(flat_elliptical_moffat_sum, (x, y), aperture.ravel(), sigma=aperture_err.ravel(), p0=guess,
                                 bounds=bounds, absolute_sigma=True)

    except RuntimeError:
        print('Unable to find fit.')
        axisarg[0][1].set_title('Fit not found within parameter bounds')
    else:

        fit_error = np.sqrt(np.diag(m_cov))
        # print('Error on parameters')
        # print(error)


        # generate a plot of fit results
        rflux1 = m_fit[0]
        rflux2 = m_fit[1]
        ra = m_fit[2]
        rb= m_fit[3]
        rbeta1 = m_fit[4]
        rbeta2 = m_fit[5]
        rx0 = m_fit[6]
        ry0 = m_fit[7]
        rtheta = m_fit[8]

        result, result_part1, result_part2 = elliptical_moffat_sum((x, y), rflux1, rflux2, ra, rb, rbeta1, rbeta2, rx0, ry0, rtheta)

        # calculate the difference between the obersved and the result fro mthe fit
        result_difference = aperture - result


        """Chi squared calculations
        """
        observed = aperture.ravel()

        # generate a plot of fit results

        expected = flat_elliptical_moffat_sum((x, y), rflux1, rflux2, ra, rb, rbeta1, rbeta2, rx0, ry0, rtheta)

        # calculated raw chi squared, including background noise
        chisq = sum(np.divide((observed - expected) ** 2, (expected + background_dev**2)))

        # degrees of freedom, 5 parameters
        degrees_of_freedom = observed.size - 7

        # normalized chi squared
        chisq_norm = chisq / degrees_of_freedom

        #print the results
        print(f'flux1: {m_fit[0]: .2f}±{fit_error[0]:.2f}')
        print(f'flux2: {m_fit[1]: .2f}±{fit_error[1]:.2f}')
        print(f'a: {m_fit[2]: .2f}±{fit_error[2]:.2f}')
        print(f'b: {m_fit[3]: .2f}±{fit_error[3]:.2f}')
        print(f'beta1: {m_fit[4]: .2f}±{fit_error[4]:.2f}')
        print(f'beta2: {m_fit[5]: .2f}±{fit_error[5]:.2f}')
        print(f'x0: {m_fit[6]: .2f}±{fit_error[6]:.2f}')
        print(f'y0: {m_fit[7]: .2f}±{fit_error[7]:.2f}')
        print(f'theta: {m_fit[8]: .2f}±{fit_error[8]:.2f}')

        print('Normalized chi squared: ')
        print(chisq_norm)

        #plot it!
        residual_im = axisarg[0][1].imshow(result_difference, norm=norm, origin='lower', cmap='viridis')
        f1.colorbar(residual_im, ax=axisarg[0][1])
        axisarg[0][1].set_title(f'Residuals. Chisq={chisq_norm:.2f}')

        # plot the separate Moffat functions
        moffat1 = axisarg[2][0].imshow(result_part1, norm=norm, origin='lower', cmap='viridis')
        f1.colorbar(moffat1, ax=axisarg[2][0])
        axisarg[2][0].set_title(f'Moffat 1: beta={rbeta1:.2f}')

        moffat2 = axisarg[2][1].imshow(result_part2, norm=norm, origin='lower', cmap='viridis')
        f1.colorbar(moffat2, ax=axisarg[2][1])
        axisarg[2][1].set_title(f'Moffat 2: beta={rbeta2:.2f}')


plt.show()
