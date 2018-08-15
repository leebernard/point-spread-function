

import pickle
import numpy as np
import matplotlib.pyplot as plt

"""
This is a script for plotting the results from fitting astronomical objects

The fit results are from fitting astronomical objects to a sum of two 
moffat functions. The data (aperture) that was used to produce each fit is 
included, as well as the location of the lower left corner of each aperture

flat_elliptical_Moffat(indata, flux, x0, y0, beta, a, b, theta):
"""
filename_list = []
# convient holder for legend
legend_list = []
# filename_list.append('/home/lee/Documents/decam-N9-A-archive.pkl')
# legend_list.append('CCD N9, amp A')
filename_list.append('/home/lee/Documents/decam-N9-B-archive.pkl')
legend_list.append('CCD N9, amp B')
filename_list.append('/home/lee/Documents/decam-N4-A-archive.pkl')
legend_list.append('CCD N4, amp A')
filename_list.append('/home/lee/Documents/decam-N4-B-archive.pkl')
legend_list.append('CCD N4, amp B')
filename_list.append('/home/lee/Documents/decam-S5-A-archive.pkl')
legend_list.append('CCD S5, amp A')
filename_list.append('/home/lee/Documents/decam-S5-B-archive.pkl')
legend_list.append('CCD S5, amp B')
filename_list.append('/home/lee/Documents/decam-N5-A-archive.pkl')
legend_list.append('CCD N5, amp A')
# filename_list.append('/home/lee/Documents/decam-N5-B-archive.pkl')
# legend_list.append('CCD N5, amp B')
filename_list.append('/home/lee/Documents/decam-N3-A-archive.pkl')
legend_list.append('CCD N3, amp A')
filename_list.append('/home/lee/Documents/decam-N3-B-archive.pkl')
legend_list.append('CCD N3, amp B')
filename_list.append('/home/lee/Documents/decam-N16-A-archive.pkl')
legend_list.append('CCD N16, amp A')
filename_list.append('/home/lee/Documents/decam-N16-B-archive.pkl')
legend_list.append('CCD N16, amp B')

# ('CCD N9, amp A', 'CCD N9, amp B', ' CCD N4, amp A', 'CCD N4, amp B', 'CCD S5, amp A', 'CCD S5, amp B', 'CCD N5, amp A','CCD N5, amp B')

# filename = '/home/lee/Documents/single-moffat-archive-im7.pkl'

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
# filename_list.append('/home/lee/Documents/single-moffat-archive-im16.pkl')

# figures for plotting
plt.figure('alpha values', figsize=(12, 10))
plt.figure('beta values', figsize=(12, 10))
plt.figure('Full Width, Half Maximum', figsize=(12, 10))
for n, filename in enumerate(filename_list):
    with open(filename, mode='rb') as file:
        archive = pickle.load(file)

    apertures = archive['apertures']
    parameters = archive['parameters']
    background = archive['background']
    cov = archive['param_cov']

    parameters = np.asarray(parameters)
    background = np.asarray(background)

    # list to hold the deviations on the parameters
    error_list = []
    ab_cov = []
    abeta_cov = []
    bbeta_cov = []
    for cov_mat in cov:
        error_list.append(np.sqrt(np.diag(cov_mat)))

        # unpack the covariance between the a and b parameters
        ab_cov.append(cov_mat[4][5])

        # unpack the covariance between the a, b and beta parameters
        abeta_cov.append(cov_mat[4][3])
        bbeta_cov.append(cov_mat[5][3])

    # convert to numpy array, for convenience
    error_list = np.asarray(error_list)
    ab_cov = np.asarray(ab_cov)
    abeta_cov = np.asarray(abeta_cov)
    bbeta_cov = np.asarray(bbeta_cov)


    # unpack the needed deviation values
    sigma_flux = error_list[:, 0]
    sigma_a = error_list[:, 4]
    sigma_b = error_list[:, 5]
    sigma_beta = error_list[:, 3]


    # unpack the measured Flux
    measured_flux = []
    max_pixel = []
    for aperture in apertures:
        measured_flux.append(np.sum(aperture))
        max_pixel.append(np.max(aperture))
    # convert to a numpy array. This is done separately, to avoid unnecessary copying of arrays
    measured_flux = np.asarray(measured_flux)
    max_pixel = np.asarray(max_pixel)

    # extract the aperture size
    # aperture_size = [apt.size for apt in apertures]
    aperture_size = []
    for apt in apertures:
        aperture_size.append(apt.size)
    # convert to array for convience
    aperture_size = np.asarray(aperture_size)

    # split background apart into values and deviation
    background, bkg_dev = background[:, 0], background[:, 1]

    # calculate signal to noise ratio
    noise = np.sqrt(measured_flux + aperture_size * background)

    sn_ratio = measured_flux/noise





    # unpack the width parameters a and b
    # unpack the beta1 parameter
    # unpack the beta2 parameter
    a_param = []
    b_param = []
    beta = []
    for m, parameter in enumerate(parameters):
        a_param.append(parameter[4])
        b_param.append(parameter[5])

        beta.append(parameter[3])
    # convert to numpy arrays
    a_param = np.asarray(a_param)
    b_param = np.asarray(b_param)
    beta = np.asarray(beta)

    # calculate the average width, alpha
    alpha = np.sqrt(a_param * b_param)
    # calculate the deviations for alpha
    sigma_alpha = .5 * np.sqrt(b_param / a_param * (sigma_a ** 2) + a_param / b_param * (sigma_b ** 2) + 2 * ab_cov)

    # calculate the Full Width, Half Max
    fwhm = 2*alpha
    sigma_fwhm = 2*sigma_alpha

    """
    create a boolean mask that clips values from the plot that have a s/n less than 60
    """
    sn_clip_mask = np.zeros(max_pixel.size, dtype=bool)
    threshold = 60
    for m, pixel in enumerate(max_pixel):
        # if relative error is above the threshold, mask the corresponding parameter results
        if sn_ratio[m] < threshold:
            sn_clip_mask[m] = True
    # apply the mask to the x values. this surpresses plotting of the value
    max_pixel = np.ma.array(max_pixel, mask=sn_clip_mask)

    # plot the stuff
    plt.figure('alpha values')  # select correct figure
    plt.errorbar(max_pixel, alpha, yerr=sigma_alpha, ls='None', marker='o', capsize=3)

    plt.figure('beta values')  # select correct figure
    plt.errorbar(max_pixel, beta, yerr=sigma_beta, ls='None', marker='o', capsize=3)

    plt.figure('Full Width, Half Maximum')
    plt.errorbar(max_pixel, fwhm, yerr=sigma_fwhm, ls='None', marker='o', capsize=3)

plt.figure('alpha values')
plt.title('Single Moffat Alpha Values(Half Width Half Max)')
plt.xlabel('Measured Flux (e-)')
plt.ylabel('Average Width alpha (pixels)')
plt.ylim(1.5, 2.5)
# plt.legend(('Frame 1', 'Frame 2', 'Frame 4', 'Frame 5', 'Frame 7', 'Frame 9', 'Frame 10', 'Frame 12', 'Frame 13', 'Frame 16'), loc='best')
plt.legend(legend_list)

plt.figure('beta values')
plt.title('Single Moffat Beta Values')
plt.ylabel(r'$\beta$ Value')
plt.xlabel('Measured Flux (e-)')
plt.ylim(3.5, 5.5)
plt.legend(legend_list)

plt.figure('Full Width, Half Maximum')
plt.title('FWHM vs Flux')
plt.xlabel('Measured Flux (e-)')
plt.ylabel('Full Width, Half Maximum (pixels)')
plt.ylim(3.5, 4.5)
plt.legend(legend_list)

plt.show()