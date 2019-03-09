
"""
This is a script for examining the full width half max of the psf fits.

The goal is to calculate the 'original' psf FWHM by finding the linear fit of the slope, and taking
the intercept, called sigma_0.  sigma_0 is then used to generate a delta_sigma/sigma_0 spread, where
delta_sigma is sigma_i - sigma_0. This is generated for each CCD in each image, and should normalize
out any differences between the images, allowing data from all the CCDs and images to be compared directly.

The percent increase of the FWHM should be the slope of the delta_sigma/sigma_0 scatter. The slope of the
delta_sigma/sigma_0 scatter is found via a linear fit. The std deviations on the fit are then calculated
by bootstrapping.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt

from astropy.stats import sigma_clip
from scipy.stats import ttest_ind

filename = '/home/lee/Documents/decam-ccds-N4-S4-20170331-unbiased-forcedangle-archive.pkl'

# open the archive
with open(filename, mode='rb') as file:
    archive_list = pickle.load(file)

# split the archived data, so that it can be paired up
archive_listA = archive_list[::2]  # takes every other entry, starting at 0
archive_listB = archive_list[1::2]  # starting at 1

legend_list = []
flux_list = []
row_delta_list = []
sigma_row_delta_list = []
col_delta_list = []
sigma_col_delta_list = []
clip_mask_list = []

for archiveA, archiveB in zip(archive_listA, archive_listB):
    # unpack archive A
    apertures = archiveA['apertures']
    background = archiveA['background']
    parameters = archiveA['parameters']
    cov = archiveA['param_cov']
    # this unpacks the names of the CCDs. Archive B contains amplifier names
    legend_list.append(archiveA['dataset_name'])

    # unpack archive B, adding it to archive A
    apertures.extend(archiveB['apertures'])
    background.extend(archiveB['background'])
    parameters.extend(archiveB['parameters'])
    cov.extend(archiveB['param_cov'])

    # convert to array
    parameters = np.asarray(parameters)
    background = np.asarray(background)

    # list to hold the deviations on the parameters
    error_list = []
    ab_cov = []
    abeta_cov = []
    bbeta_cov = []
    # 95% confidence interval for 6 parameters
    delta_chisqrd = 4
    for cov_mat in cov:
        error_list.append(np.sqrt(np.diag(cov_mat) * delta_chisqrd))

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
    sigma_theta = error_list[:, -1]

    # unpack the measured Flux
    measured_flux = []
    max_pixel = []
    for aperture in apertures:
        measured_flux.append(np.sum(aperture))
        max_pixel.append(np.max(aperture))
    # convert to a numpy array. This is done separately, to avoid unnecessary copying of arrays
    measured_flux = np.asarray(measured_flux)
    max_pixel = np.asarray(max_pixel)

    # find the S/N ratio
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
    # unpack the beta parameter
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


    # clip any values that have a S/N ration below the threshold
    sn_clip_mask = np.zeros(measured_flux.size, dtype=bool)
    threshold = 100
    for n, _ in enumerate(measured_flux):
        # if relative error is above the threshold, mask the corresponding parameter results
        if sn_ratio[n] < threshold or measured_flux[n]>5000000:
            sn_clip_mask[n] = True
    # apply the mask to the x values. this surpresses plotting of the value
    max_pixel = np.ma.array(max_pixel, mask=sn_clip_mask)
    masked_flux = np.ma.array(measured_flux, mask=sn_clip_mask)

    # fit the widths to a linear functions
    # parameter 'a' corresponds to x direction, which corresponds to rows
    poly_coeffs_a, poly_cov_a = np.ma.polyfit(masked_flux, a_param, deg=1, w=1/sigma_a, cov=True)
    # parameter 'b' corresponds to y direction, which corresponds to columns
    poly_coeffs_b, poly_cov_b = np.ma.polyfit(masked_flux, b_param, deg=1, w=1/sigma_b, cov=True)

    # unpack the wanted parameters
    row_width_0 = poly_coeffs_a[-1]
    print('row fit results:', poly_coeffs_a)
    col_width_0 = poly_coeffs_b[-1]
    print('column fit results:', poly_coeffs_b)

    # unpack the parameter errors
    # 95% confidence for 1 degree of freedom
    delta_chisqrd = 4
    sigma_row_width_0 = delta_chisqrd * poly_cov_a[-1][-1]
    sigma_col_width_0 = delta_chisqrd * poly_cov_b[-1][-1]

    # calculate the delta sigma over sigma
    delta_hfhm_row = (a_param - row_width_0)/row_width_0
    delta_hfhm_col = (b_param - col_width_0)/col_width_0
    # and the errors
    sigma_delta_hfhm_row = np.sqrt(sigma_row_width_0**2 + sigma_a**2)
    sigma_delta_hfhm_col = np.sqrt(sigma_col_width_0**2 + sigma_b**2)

    # store the needed data
    flux_list.extend(measured_flux)
    clip_mask_list.extend(sn_clip_mask)
    row_delta_list.extend(delta_hfhm_row)
    sigma_row_delta_list.extend(sigma_delta_hfhm_row)
    col_delta_list.extend(delta_hfhm_col)
    sigma_col_delta_list.extend(sigma_delta_hfhm_col)

# mask the unwanted values, convert to array
flux_values = np.ma.array(flux_list, mask=clip_mask_list)
row_delta_list = np.asarray(row_delta_list)
col_delta_list = np.asarray(col_delta_list)

bf_row_slope, bf_row_intercept = np.ma.polyfit(flux_values, row_delta_list, deg=1)
bf_col_slope, bf_col_intercept = np.ma.polyfit(flux_values, col_delta_list, deg=1)

# generate std deviations through bootstrapping
iters = 10000
row_sample_fits = []
col_sample_fits = []
for _ in range(iters):

    # generate an array of integers that correspond to the size of the data set
    indexarray = np.arange(len(flux_values))

    # sample the array locations
    # if replace is True, points in parent population can be reused in sample
    sample_indexs = np.random.choice(indexarray, size=len(indexarray), replace=True)

    # generate fit

    row_sample_fits.append(np.ma.polyfit(flux_values[sample_indexs], row_delta_list[sample_indexs], deg=1))
    col_sample_fits.append(np.ma.polyfit(flux_values[sample_indexs], col_delta_list[sample_indexs], deg=1))

sigma_row_slope, sigma_row_intercept = np.std(np.asarray(row_sample_fits), axis=0)
sigma_col_slope, sigma_col_intercept = np.std(np.asarray(col_sample_fits), axis=0)

# make a histogram of the results
row_slope_distro, row_intercept_distro = np.hsplit(np.asarray(row_sample_fits), 2)
col_slope_distro, col_intercept_distro = np.hsplit(np.asarray(col_sample_fits), 2)
bootstrap_hist = plt.figure('bootstrap fits', figsize=(12,10))
bootstrap_hist.suptitle('Bootstrapped Parameter Distrubutions', fontsize=12)
row_slope_hist = bootstrap_hist.add_subplot(221)
col_slope_hist = bootstrap_hist.add_subplot(222)
row_intercept_hist = bootstrap_hist.add_subplot(223)
col_intercept_hist = bootstrap_hist.add_subplot(224)


row_slope_hist.hist(row_slope_distro, bins=np.linspace(.2e-8, 1.1e-8, num=101), color='tab:blue')
row_slope_hist.set_title(r'$\Delta \sigma / \sigma$ Slope, Rows (1/e')

row_intercept_hist.hist(row_intercept_distro, bins=np.linspace(-.003, .007, num=101), color='tab:blue')
row_intercept_hist.set_title(r'$\Delta \sigma / \sigma$ intercept, Rows (unitless)')

col_slope_hist.hist(col_slope_distro, bins=np.linspace(.2e-8, 1.1e-8, num=101), color='tab:purple')
col_slope_hist.set_title(r'$\Delta \sigma / \sigma$ Slope, Columns (1/e')

col_intercept_hist.hist(col_intercept_distro, bins=np.linspace(-.003, .007, num=101), color='tab:purple')
col_intercept_hist.set_title(r'$\Delta \sigma / \sigma$ Intercept, Columns (unitless)')

# take a student t-test. This is preliminary
tstat, pvalue = ttest_ind(row_slope_distro, col_slope_distro)

print('Brighter-Fatter Parameters')
print(f'row: {bf_row_slope:.2e} ±{sigma_row_slope:.1e} * x + {bf_row_intercept:.1e} ±{sigma_row_intercept:.1e}')
print(f'col: {bf_col_slope:.2e} ±{sigma_col_slope:.1e} * x + {bf_col_intercept:.1e} ±{sigma_col_intercept:.1e}')
# generate values
# calcualte upper and lower values to a confidence of 95.4%
z = 2
x_values = np.arange(0, max(flux_values))
row_y_values = bf_row_slope*x_values + bf_row_intercept
row_lower_y_values = (bf_row_slope - z*sigma_row_slope)*x_values + (bf_row_intercept - z*sigma_row_intercept)
row_upper_y_values = (bf_row_slope + z*sigma_row_slope)*x_values + (bf_row_intercept + z*sigma_row_intercept)
col_y_values = bf_col_slope*x_values + bf_col_intercept
col_lower_y_values = (bf_col_slope - z*sigma_col_slope)*x_values + (bf_col_intercept - z*sigma_col_intercept)
col_upper_y_values = (bf_col_slope + z*sigma_col_slope)*x_values + (bf_col_intercept + z*sigma_col_intercept)

# plot the delta_sigma/sigma_0
plt.figure('delta_sigma/sigma_0 row width', figsize=(12, 10))
# plt.scatter(max_pixel, delta_hfhm_normalized)
plt.errorbar(flux_values, row_delta_list, yerr=sigma_row_delta_list, capsize=3, color='blue', ls='None', marker='o')
plt.plot(x_values, row_y_values, color='blue')
plt.plot(x_values, row_lower_y_values, color='blue', ls='--')
plt.plot(x_values, row_upper_y_values, color='blue', ls='--')

# plot the max pixel as a function of measured flux
plt.figure('delta_sigma/sigma_0 column width', figsize=(12, 10))
plt.errorbar(flux_values, col_delta_list, yerr=sigma_col_delta_list, capsize=3, color='purple', ls='None', marker='x', ms=6)
plt.plot(x_values, col_y_values, color='purple')
plt.plot(x_values, col_lower_y_values, color='purple', ls='--')
plt.plot(x_values, col_upper_y_values, color='purple', ls='--')

# plot both together
plt.figure('delta_sigma over sigma_0', figsize=(12, 10))
plt.errorbar(flux_values, row_delta_list, yerr=sigma_row_delta_list, capsize=3, color='blue', ls='None', marker='o')
plt.plot(x_values, row_y_values, color='blue')
plt.plot(x_values, row_lower_y_values, color='blue', ls='--')
plt.plot(x_values, row_upper_y_values, color='blue', ls='--')
plt.errorbar(flux_values, col_delta_list, yerr=sigma_col_delta_list, capsize=3, color='purple', ls='None', marker='x', ms=6)
plt.plot(x_values, col_y_values, color='purple')
plt.plot(x_values, col_lower_y_values, color='purple', ls='--')
plt.plot(x_values, col_upper_y_values, color='purple', ls='--')


plt.figure('delta_sigma/sigma_0 row width')
plt.title('Delta Sigma over Sigma in the Row Direction')
plt.xlabel('Measured Flux (e-)')
plt.ylabel('percent change in HWHM (%)')
plt.ylim(-.02, .09)
plt.legend(('Row direction widths', f'{z}-sigma boundry'), loc='best')

plt.figure('delta_sigma/sigma_0 column width')
plt.title('Delta Sigma over Sigma in the Column Direction')
plt.xlabel('Measured Flux (e-)')
plt.ylabel('percent change in HWHM (%)')
plt.ylim(-.02, .09)
plt.legend(('Row direction widths', f'{z}-sigma boundry'), loc='best')

plt.figure('delta_sigma over sigma_0')
plt.title('Delta Sigma over Sigma_0')
plt.xlabel('Measured Flux (e-)')
plt.ylabel('percent change in HWHM (%)')
plt.ylim(ymax=0.09)
plt.legend(('Row direction widths best fit', f'Row {z}-sigma boundry', '_nolegend_', 'Col direction widths best fit', f'Col {z}-sigma boundry'), loc='best')

plt.show()

