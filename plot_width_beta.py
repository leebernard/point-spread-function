

import pickle
import numpy as np
import matplotlib.pyplot as plt

"""
This is a script for plotting the results from fitting astronomical objects

The fit results are from fitting astronomical objects to a sum of two 
moffat functions. The data (aperture) that was used to produce each fit is 
included, as well as the location of the lower left corner of each aperture

"""

# load the data
filename_list = []

filename_list.append('/home/lee/Documents/sample-archive-im1.pkl')
filename_list.append('/home/lee/Documents/sample-archive-im2.pkl')
filename_list.append('/home/lee/Documents/sample-archive-im4.pkl')
filename_list.append('/home/lee/Documents/sample-archive-im5.pkl')
filename_list.append('/home/lee/Documents/sample-archive-im7.pkl')

filename_list.append('/home/lee/Documents/sample-archive-im9.pkl')
filename_list.append('/home/lee/Documents/sample-archive-im10.pkl')
filename_list.append('/home/lee/Documents/sample-archive-im12.pkl')
filename_list.append('/home/lee/Documents/sample-archive-im13.pkl')
filename_list.append('/home/lee/Documents/sample-archive-im16.pkl')

# figure for ploting the flux ratios
alpha_fig = plt.figure('alpha values', figsize=(12, 10))
beta_fig = plt.figure('beta values', figsize=(12,10))
beta_ratio_fig = plt.figure('beta ratio', figsize=(12,10))
beta_major_plot = beta_fig.add_subplot(211)
beta_minor_plot = beta_fig.add_subplot(212)
fwhm_fig = plt.figure('full width half max', figsize=(12, 10))
fwhm_beta_major_plot = fwhm_fig.add_subplot(211)
plt.title('Large beta Value')
fwhm_beta_minor_plot = fwhm_fig.add_subplot(212)
plt.title('Small beta Value')

for n, filename in enumerate(filename_list):
    with open(filename, mode='rb') as file:
        archive = pickle.load(file)


    apertures = archive['apertures']
    parameters = archive['parameters']
    cov = archive['param_cov']

    # list to hold the deviations on the parameters
    error_list = []
    ab_cov = []
    beta_cov = []
    abeta_major_cov = []
    abeta_minor_cov = []
    bbeta_major_cov = []
    bbeta_minor_cov = []
    for cov_mat in cov:
        error_list.append(np.sqrt(np.diag(cov_mat)))

        # unpack the covariance between the a and b parameters
        ab_cov.append(cov_mat[2][3])
        # unpack the covariance between beta1 and beta2
        beta_cov.append(cov_mat[4][5])
        # unpack the covariance for a, b and beta1
        abeta_major_cov.append(cov_mat[2][4])
        bbeta_major_cov.append(cov_mat[3][4])
        # unpack the covariance for a, b and beta2
        abeta_minor_cov.append(cov_mat[2][5])
        bbeta_minor_cov.append(cov_mat[3][5])
    # convert to numpy array, for convenience
    error_list = np.asarray(error_list)
    ab_cov = np.asarray(ab_cov)
    beta_cov = np.asarray(beta_cov)
    abeta_major_cov = np.asarray(abeta_major_cov)
    abeta_minor_cov = np.asarray(abeta_minor_cov)
    bbeta_major_cov = np.asarray(bbeta_major_cov)
    bbeta_minor_cov = np.asarray(bbeta_minor_cov)

    # unpack the needed deviation values
    sigma_flux = error_list[:, 0]
    sigma_a = error_list[:, 2]
    sigma_b = error_list[:, 3]
    sigma_beta_major = error_list[:, 4]
    sigma_beta_minor = error_list[:, 5]

    # unpack the calculated Flux
    measured_flux = []
    for aperture in apertures:
        measured_flux.append(np.sum(aperture))
    # convert to a numpy array. This is done separately, to avoid unnecessary copying of arrays
    measured_flux = np.asarray(measured_flux)

    # unpack the calculated flux
    calc_flux = []
    for parameter in parameters:
        calc_flux.append(parameter[0])
    # convert to numpy array
    calc_flux = np.asarray(calc_flux)

    # unpack the width parameters a and b
    # unpack the beta1 parameter
    # unpack the beta2 parameter
    a_param = []
    b_param = []
    beta_major = []
    beta_minor = []
    for m, parameter in enumerate(parameters):
        a_param.append(parameter[2])
        b_param.append(parameter[3])

        # check if beta1 and two got flipped
        # beta1 (beta major) should always be larger than beta2 (beta minor)
        if parameter[4] < parameter[5]:
            beta_major.append(parameter[5])
            beta_minor.append(parameter[4])
            # swap the error values
            sigma_beta_major[m], sigma_beta_minor[m] = sigma_beta_minor[m], sigma_beta_major[m]
            # swap the covariance values
            abeta_major_cov, abeta_minor_cov = abeta_minor_cov, abeta_major_cov
            bbeta_major_cov, bbeta_minor_cov = bbeta_minor_cov, bbeta_major_cov
        else:
            beta_major.append(parameter[4])
            beta_minor.append(parameter[5])
    # convert to numpy arrays
    a_param = np.asarray(a_param)
    b_param = np.asarray(b_param)
    beta_major = np.asarray(beta_major)
    beta_minor = np.asarray(beta_minor)


    # calculate the average width, alpha
    alpha = np.sqrt(a_param * b_param)
    # calculate the deviations for alpha
    sigma_alpha = .5*np.sqrt(b_param/a_param*(sigma_a**2) + a_param/b_param*(sigma_b**2) + 2*ab_cov)

    # calculate the Full Width, Half Max for Beta Major
    fwhm_major = 2*alpha*np.sqrt(2**(1/beta_major) - 1)
    varience_fwhm_major = ((np.sqrt(2**(1/beta_major) - 1))*sigma_alpha)**2 \
        + (np.divide(alpha*2**(1/beta_major)*np.log(1/2), (2*np.sqrt(2**(1/beta_major) - 1)*beta_major**2))*sigma_beta_major)**2 \
        + (np.sqrt(b_param/a_param)*abeta_major_cov + np.sqrt(a_param/b_param)*bbeta_major_cov) * (alpha*np.log(1/2)*2**(1/beta_major))/(2*beta_major**2)
    sigma_fwhm_major = 2*np.sqrt(varience_fwhm_major)

    # calculate the Full Width, Half Max for Beta Minor
    fwhm_minor = 2*alpha*np.sqrt(2**(1/beta_minor) - 1)
    varience_fwhm_minor = ((np.sqrt(2**(1/beta_minor) - 1))*sigma_alpha)**2 \
        + (np.divide(alpha*2**(1/beta_minor)*np.log(1/2), (2*np.sqrt(2**(1/beta_minor) - 1)*beta_minor**2))*sigma_beta_minor)**2 \
        + (np.sqrt(b_param/a_param)*abeta_minor_cov + np.sqrt(a_param/b_param)*bbeta_minor_cov) * (alpha*np.log(1/2)*2**(1/beta_minor))/(2*beta_minor**2)
    sigma_fwhm_minor = 2*np.sqrt(varience_fwhm_minor)

    # calculate the relative errors
    relative_alpha = sigma_alpha/alpha
    relative_flux = sigma_flux/calc_flux
    relative_beta_major = sigma_beta_major/beta_major
    relative_beta_minor = sigma_beta_minor/beta_minor

    # reject any parameters with relative errors above a certain amount
    mask = np.zeros(alpha.shape, dtype=bool)
    threshold = 1
    for m, tuple in enumerate(zip(relative_alpha, relative_flux, relative_beta_major, relative_beta_minor)):
        # if relative error is above the threshold, mask the corresponding parameter results
        if any(np.array(tuple) > threshold):
            mask[m] = True
    # apply the mask to the x values. this surpresses plotting of the value
    measured_flux = np.ma.array(measured_flux, mask=mask)

    # calculate the beta ratios and deviation
    beta_ratio = np.divide(beta_major, beta_minor)
    beta_ratio_dev = np.sqrt(sigma_beta_major**2 + sigma_beta_minor**2 - 2*beta_cov)

    # calculate the Full Width, Half Max


    # plot the stuff
    plt.figure('alpha values')  # select correct figure
    plt.errorbar(measured_flux, alpha, yerr=sigma_alpha, ls='None', marker='o', capsize=3)

    plt.figure('beta values')  # select correct figure
    beta_major_plot.errorbar(measured_flux, beta_major, yerr=sigma_beta_major, ls='None', marker='o', capsize=3)
    beta_minor_plot.errorbar(measured_flux, beta_minor, yerr=sigma_beta_minor, ls='None', marker='o', capsize=3)

    plt.figure('beta ratio')
    # plt.errorbar(measured_flux, beta_ratio, yerr=beta_ratio_dev, ls='None', marker='o', capsize=3)
    plt.plot(measured_flux, beta_ratio, ls='None', marker='o')

    plt.figure('full width half max')
    fwhm_beta_major_plot.errorbar(measured_flux, fwhm_major, yerr=sigma_fwhm_major, ls='None', marker='o', capsize=3)
    fwhm_beta_minor_plot.errorbar(measured_flux, fwhm_minor, yerr=sigma_fwhm_minor, ls='None', marker='o', capsize=3)

plt.figure('alpha values')
plt.xlabel('Measured Flux (e-)')
plt.ylabel('Average Width alpha (pixels)')
plt.legend(('Frame 1', 'Frame 2', 'Frame 4', 'Frame 5', 'Frame 7', 'Frame 9', 'Frame 10', 'Frame 12', 'Frame 13', 'Frame 16'), loc='best')

plt.figure('beta values')
beta_major_plot.set_ylim(0, 22)
beta_major_plot.set_ylabel(r'Major $\beta$ Value')
beta_minor_plot.set_ylim(0, 22)
beta_minor_plot.set_xlabel('Measured Flux (e-)')
beta_minor_plot.set_ylabel(r'Minor $\beta$ Value')

plt.figure('beta ratio')
plt.xlabel('Measured Flux (e-)')
plt.title(r'Ratio of major $\beta$ value to minor $\beta$ value')
plt.ylim(0, 6)

plt.figure('full width half max')
fwhm_beta_major_plot.set_ylabel('Full Width Half Maximum (pixels)')
fwhm_beta_minor_plot.set_ylabel('Full Width Half Maxiumum (pixels)')
fwhm_beta_minor_plot.set_xlabel('Measured Flux (e-)')

plt.show()

