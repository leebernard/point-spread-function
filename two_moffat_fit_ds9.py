

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



# generate each curve fit
from scipy.optimize import curve_fit
for aperture in aperture_list:

    print('---------------------')
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


    # generate a best guess
    y_guess = aperture.shape[0] / 2
    x_guess = aperture.shape[1] / 2
    flux1_guess = np.sum(aperture)*0.8
    flux2_guess = np.sum(aperture)*0.2
    beta1_guess = 7
    beta2_guess = 2
    alpha_guess = 4

    guess = [flux1_guess, flux2_guess, alpha_guess, beta1_guess, beta2_guess, x_guess, y_guess]

    # indexes of the apature, remembering that python indexes vert, horz
    y = np.arange(aperture.shape[0])
    x = np.arange(aperture.shape[1])
    x, y = np.meshgrid(x, y)


    # curve fit
    try:
        m_fit, m_cov = curve_fit(flat_Moffat_sum, (x, y), aperture.ravel(), sigma=aperture_err.ravel(), p0=guess)

    except RuntimeError:
        print('Unable to find fit.')
        axisarg[0][1].set_title('Fit not found within parameter bounds')
    else:

        error = np.sqrt(np.diag(m_cov))
        # print('Error on parameters')
        # print(error)


        # generate a plot of fit results
        rflux1 = m_fit[0]
        rflux2 = m_fit[1]
        ralpha = m_fit[2]
        rbeta1 = m_fit[3]
        rbeta2 = m_fit[4]
        rx0 = m_fit[5]
        ry0 = m_fit[6]

        result, result_part1, result_part2 = Moffat_sum((x, y), rflux1, rflux2, ralpha, rbeta1, rbeta2, rx0, ry0)

        # calculate the difference between the obersved and the result fro mthe fit
        result_difference = aperture - result



        """Chi squared calculations"""
        """Chi squared calculations
        """
        observed = aperture.ravel()

        m_input = (x, y)
        flux1 = result[0]
        flux2 = result[1]
        alpha = result[2]
        beta1 = result[3]
        beta2 = result[4]
        x0 = result[5]
        y0 = result[6]

        expected = flat_Moffat_sum(m_input, flux1, flux2, alpha, beta1, beta2, x0, y0)

        # calculated raw chi squared, including background noise
        chisq = sum(np.divide((observed - expected) ** 2, (expected + background_dev**2)))

        # degrees of freedom, 5 parameters
        degrees_of_freedom = observed.size - 7

        # normalized chi squared
        chisq_norm = chisq / degrees_of_freedom

        #print the results
        print('Resultant parameters')
        print(f'Flux1: {rflux1:.2f} ± {error[0]:.2f}')
        print(f'Flux2: {rflux2:.2f} ± {error[1]:.2f}')
        print(f'Center (x, y): {rx0:.2f} ± {error[5]:.2f}, {ry0:.2f} ± {error[6]:.2f}')
        print(f'alpha: {ralpha:.2f} ± {error[2]:.2f}')
        print(f'beta1: {rbeta1:.2f} ± {error[3]:.2f}')
        print(f'beta2: {rbeta2:.2f} ± {error[4]:.2f}')

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
