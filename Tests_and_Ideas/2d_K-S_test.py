"""
Two-dimensional Kolmogorov-Smirnov test of sample to a model. This is
translated from c code, for practice.
"""

import pickle
import numpy as np
# import matplotlib.pyplot as plt

from scipy.stats import ks_2samp
from scipy.stats import pearsonr
from astropy.stats import sigma_clip

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
row_delta_values = np.ma.array(row_delta_list, mask=clip_mask_list)
col_delta_values = np.ma.array(col_delta_list, mask=clip_mask_list)

print('Correlation between row width and flux:', pearsonr(flux_values, row_delta_values))
print('Correlation between col width and flux:', pearsonr(flux_values, col_delta_values))
print('-----------------------------------------')
r, r_pvalue = pearsonr(row_delta_values, col_delta_values)
print('Correlation coeff between the sets:', r)
print('p-value of correlation coeff:', r_pvalue)
print('-----------------------------------------')
prob, pvalue = ks_2samp(row_delta_values, col_delta_values)

print('probability of nul hyp producing this spread:', pvalue)
print('KS stat:', prob)