

import pickle
import numpy as np
import matplotlib.pyplot as plt

from scipy.special import gammaincc

"""
This is a script for plotting the results from fitting astronomical objects

The fit results are from fitting astronomical objects to a sum of two
moffat functions. The data (aperture) that was used to produce each fit is
included, as well as the location of the lower left corner of each aperture

"""



def flat_elliptical_Moffat(indata, flux, x0, y0, beta, a, b, theta):
    """Model of PSF using a single Moffat distribution, with elliptical parameters.

    Includes a parameter for  axial alignment. This function flattens the
    output, for curve fitting.

    Parameters
    ----------
    indata: list
        a list of 2 arrays. The first array is the x values per data point. The
        second array is the y values per data point.
    flux: float
        Represents the total flux of the object
    x0: float
        horizontal location of the centroid
    y0: float
        vertical location of the centroid
    beta: float
        change in slope parameter
    a: float
        width parameter in the x direction
    b: float
        width parameter in the y direction
    theta: float
        angle of eccentricity
    offset: float
        estimate of background. Should be zero

    Returns
    -------
    moffat_fun.ravel(): flattened array-like
        array of data values produced from the x and y inputs. Flattened, for
        curve fitting
    """
    x_in, y_in = indata

    # moffat_fun = offset + flux * normalize * (1 + ((x - x0)**2/a**2 + (y - y0)**2/b**2))**(-beta)
    A = np.cos(theta) ** 2 / a ** 2 + np.sin(theta) ** 2 / b ** 2
    B = 2 * np.cos(theta) * np.sin(theta) * (1 / a ** 2 - 1 / b ** 2)
    C = np.sin(theta) ** 2 / a ** 2 + np.cos(theta) ** 2 / b ** 2

    def moffat_fun(x, y): return (1 + (A*(x - x0)**2 + B*(x - x0)*(y - y0) + C*(y - y0)**2)*(2**(1/beta) - 1))**(-beta)

    # numerical normalization
    # scale steps according to the size of the array.

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
    normalize = np.sum(moffat_fun(x_step, y_step))*delta_x*delta_y
    # normalize = 1

    # forget that, just integrate it
    # normalize, norm_err = dblquad(moffat_fun, -np.inf, np.inf, lambda x: -np.inf, lambda x: np.inf)

    output = flux*moffat_fun(x_in, y_in)/normalize

    return output.ravel()



# filename_listA = []
# filename_listB = []
# legend_list = []

filename = '/home/lee/Documents/decam-ccds-N4-S4-20170331-unbiased-archive.pkl'
# filename_listA.append('/home/lee/Documents/decman-fit-archive-20170331/decam-94s-S4-A-archive.pkl')
# legend_list.append('S4 2:48UT-94s')
# filename_listB.append('/home/lee/Documents/decman-fit-archive-20170331/decam-94s-S4-B-archive.pkl')
# # legend_list.append('94s-S4-B')
# filename_listA.append('/home/lee/Documents/decman-fit-archive-20170331/decam-94s-N4-A-archive.pkl')
# legend_list.append('N4 2:48UT-94s')
# filename_listB.append('/home/lee/Documents/decman-fit-archive-20170331/decam-94s-N4-B-archive.pkl')
# # legend_list.append('94s-N4')
#
# filename_listA.append('/home/lee/Documents/decman-fit-archive-20170331/decam-91s-S4-A-archive.pkl')
# legend_list.append('S4 2:16UT-91s')
# filename_listB.append('/home/lee/Documents/decman-fit-archive-20170331/decam-91s-S4-B-archive.pkl')
# # legend_list.append('91s-S4-B')
# filename_listA.append('/home/lee/Documents/decman-fit-archive-20170331/decam-91s-N4-A-archive.pkl')
# legend_list.append('N4 2:16UT-91s')
# filename_listB.append('/home/lee/Documents/decman-fit-archive-20170331/decam-91s-N4-B-archive.pkl')
# # legend_list.append('91s-N4-B')
#
# filename_listA.append('/home/lee/Documents/decman-fit-archive-20170331/decam-102s-S4-A-archive.pkl')
# legend_list.append('S4 2:11UT-102s')
# filename_listB.append('/home/lee/Documents/decman-fit-archive-20170331/decam-102s-S4-B-archive.pkl')
# # legend_list.append('102s-S4-B')
# filename_listA.append('/home/lee/Documents/decman-fit-archive-20170331/decam-102s-N4-A-archive.pkl')
# legend_list.append('N4 2:11UT-102s')
# filename_listB.append('/home/lee/Documents/decman-fit-archive-20170331/decam-102s-N4-B-archive.pkl')
# # legend_list.append('102s-N4-B'

# filename_list.append('/home/lee/Documents/decam-94s-S4-A-archive.pkl')
# filename_list.append('/home/lee/Documents/decam-94s-S4-B-archive.pkl')
# filename_list.append('/home/lee/Documents/decam-94s-N4-A-archive.pkl')
# filename_list.append('/home/lee/Documents/decam-94s-N4-B-archive.pkl')
#
# filename_list.append('/home/lee/Documents/decam-102s-S4-A-archive.pkl')
# filename_list.append('/home/lee/Documents/decam-102s-S4-B-archive.pkl')
# filename_list.append('/home/lee/Documents/decam-102s-N4-A-archive.pkl')
# filename_list.append('/home/lee/Documents/decam-102s-N4-B-archive.pkl')
#
# filename_list.append('/home/lee/Documents/decam-91s-S4-A-archive.pkl')
# filename_list.append('/home/lee/Documents/decam-91s-S4-B-archive.pkl')
# filename_list.append('/home/lee/Documents/decam-91s-N4-A-archive.pkl')
# filename_list.append('/home/lee/Documents/decam-91s-N4-B-archive.pkl')

# filename_list.append('/home/lee/Documents/decam-N9-A-archive.pkl')
# filename_list.append('/home/lee/Documents/decam-N9-B-archive.pkl')
# filename_list.append('/home/lee/Documents/decam-N4-A-archive.pkl')
# filename_list.append('/home/lee/Documents/decam-N4-B-archive.pkl')
# filename_list.append('/home/lee/Documents/decam-S5-A-archive.pkl')
# filename_list.append('/home/lee/Documents/decam-S5-B-archive.pkl')
# filename_list.append('/home/lee/Documents/decam-N5-A-archive.pkl')
# filename_list.append('/home/lee/Documents/decam-N5-B-archive.pkl')
# filename_list.append('/home/lee/Documents/decam-N3-A-archive.pkl')
# filename_list.append('/home/lee/Documents/decam-N3-B-archive.pkl')

# filename_list.append('/home/lee/Documents/single-moffat-archive-im1.pkl')
# filename_list.append('/home/lee/Documents/single-moffat-archive-im2.pkl')
# filename_list.append('/home/lee/Documents/single-moffat-archive-im4.pkl')
# filename_list.append('/home/lee/Documents/single-moffat-archive-im5.pkl')
# filename_list.append('/home/lee/Documents/single-moffat-archive-im7.pkl')
#
# filename_list.append('/home/lee/Documents/single-moffat-archive-im9.pkl')
# filename_list.append('/home/lee/Documents/single-moffat-archive-im10.pkl')
# filename_list.append('/home/lee/Documents/single-moffat-archive-im12.pkl')
# filename_list.append('/home/lee/Documents/single-moffat-archive-im13.pkl')
# filename_list.append('/home/lee/Documents/sample-single-moffat-im16.pkl')

# open the archive
with open(filename, mode='rb') as file:
    archive_list = pickle.load(file)

# split the archived data, so that it can be paired up
archive_listA = archive_list[::2]
archive_listB = archive_list[1::2]

# list for storing relative error values
relative_err_store = []
# list for storing the legend names
legend_list = []
plt.figure('Error vs sn ratio', figsize=(12, 10))
plt.figure('Goodness of fit vs SN ratio', figsize=(12, 10))
plt.figure('Goodness of fit vs Peak Pixel Value', figsize=(12, 10))
for n, archive in enumerate(archive_listA):
    # with open(filename, mode='rb') as file:
    #     archive = pickle.load(file)
    apertures = archive['apertures']
    background = archive['background']
    parameters = archive['parameters']
    cov = archive['param_cov']
    legend_list.append(archive['dataset_name'])
    # add the second half of the CCD
    # with open(filename_listB[n], mode='rb') as file:
    #     archive = pickle.load(file)
    archiveB = archive_listB[n]
    apertures.extend(archiveB['apertures'])
    parameters.extend(archiveB['parameters'])
    background.extend(archiveB['background'])
    cov.extend(archiveB['param_cov'])

    # convert to np array for convience
    parameters = np.asarray(parameters)
    background = np.asarray(background)

    # list to hold the deviations on the parameters
    errors = []
    for cov_mat in cov:
        errors.append(np.sqrt(np.diag(cov_mat)))
    # convert to numpy array, for convenience
    errors = np.asarray(errors)

    # split background apart into values and deviation
    background, bkg_dev = background[:, 0], background[:, 1]

    # extract the aperture size
    # aperture_size = [apt.size for apt in apertures]
    aperture_size = []
    for apt in apertures:
        aperture_size.append(apt.size)
    # convert to array for convience
    aperture_size = np.asarray(aperture_size)

    # unpack the measured flux
    measured_flux = []
    for aperture in apertures:
        measured_flux.append(np.sum(aperture))
    # convert to a numpy array. This is done separately, to avoid unnecessary copying of arrays
    measured_flux = np.asarray(measured_flux)

    # calculate signal to noise ratio
    noise = np.sqrt(measured_flux + aperture_size * background)
    ratio = measured_flux/noise

    # extract the relative error, defined as deviation/value
    relative_err = errors/parameters
    # add the relative error to the storage list, for the histogram
    relative_err_store.extend(relative_err[:, 0:6].flatten())

    # plot the ratio of flux to background vs error


    plt.figure('Error vs sn ratio')
    plt.plot(ratio, relative_err[:,0], ls='None', marker='v', markersize=10, color='tab:blue')  # , label='Flux')
    plt.plot(ratio, relative_err[:,4], ls='None', marker='o', color='tab:red')  # , label='a')
    plt.plot(ratio, relative_err[:,5], ls='None', marker='+', markersize=12, color='tab:green')  # , label='b')
    plt.plot(ratio, relative_err[:,3], ls='None', marker='o', color='tab:purple')  # , label='beta')

    """Chi squared calculations"""
    chisq_list = []
    chisq_norm_list = []
    goodness_fit_list = []
    peak_value = []
    for n, aperture in enumerate(apertures):
        observed = aperture.ravel()

        # store the peak pixel value
        peak_value.append(aperture.max())

        y = np.arange(aperture.shape[0])
        x = np.arange(aperture.shape[1])
        x, y = np.meshgrid(x, y)

        expected = flat_elliptical_Moffat((x, y), parameters[n][0], parameters[n][1], parameters[n][2],
                                          parameters[n][3], parameters[n][4], parameters[n][5], parameters[n][6])

        # calculated raw chi squared
        chisq = sum(np.divide((observed - expected)**2, expected + bkg_dev[n]**2))

        # degrees of freedom, 7 parameters
        degrees_of_freedom = observed.size - 7

        # normalized chi squared
        chisq_norm = chisq/degrees_of_freedom

        # probability that the discrepancies are random
        goodness_fit = gammaincc(.5*degrees_of_freedom, .5*chisq)

        # store the result
        chisq_list.append(chisq)
        chisq_norm_list.append(chisq_norm)
        goodness_fit_list.append(goodness_fit)

    # plot it
    plt.figure('Goodness of fit vs SN ratio')
    plt.plot(ratio, chisq_norm_list, ls='None', marker='o')

    plt.figure('Goodness of fit vs Peak Pixel Value')
    plt.plot(peak_value, chisq_norm_list, ls='None', marker='o')


plt.figure('Error vs sn ratio')
# plt.yscale('log')
# plt.ylim(ymax=2)
plt.ylim(0, .25)
# plt.xlim(0, 530)
plt.title('Error vs SN DECam data MOSAIC-3')
plt.xlabel('Signal to Noise Ratio')
plt.ylabel('Relative Error of Fit Parameters (Linear Scale)')
plt.legend(('Flux', 'a', 'b', 'beta',), loc='best')

plt.figure('Goodness of fit vs SN ratio')
plt.xlabel('Signal to Noise Ratio')
plt.ylabel('Normalized Chi Squared')
plt.ylim(ymax=5.5)
plt.legend(legend_list, loc='best')

plt.figure('Goodness of fit vs Peak Pixel Value')
plt.xlabel('Peak Pixel Value')
plt.ylabel('Normalized Chi Squared')
plt.ylim(ymax=5.5)
plt.legend(legend_list, loc='best')

f2 = plt.figure('Error histogram')
hist_data = np.asarray(relative_err_store)
bins = np.linspace(0, .25, num=151)
plt.hist(np.clip(hist_data, bins[0], bins[-1]), bins=bins)

plt.show()

# np.clip(values_A, bins[0], bins[-1])
#
# xdata = np.arange(20.1, 215, .25)
# ydata = 8.52/(xdata - 8.33) + .125
#
# plt.figure('Error vs sn ratio')
# plt.plot(xdata, ydata, color='tab:cyan')
#
