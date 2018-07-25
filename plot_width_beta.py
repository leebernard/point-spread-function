

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
    for cov_mat in cov:
        error_list.append(np.sqrt(np.diag(cov_mat)))

        # unpack the covariance between the a and b parameters
        ab_cov.append(cov_mat[2][3])
        # unpack the covariance between beta1 and beta2
        beta_cov.append(cov_mat[4][5])
    # convert to numpy array, for convenience
    error_list = np.asarray(error_list)
    ab_cov = np.asarray(ab_cov)
    beta_cov = np.asarray(beta_cov)

    #unpack the needed deviation values
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

    # unpack the alpha parameter
    # unpack the beta1 parameter
    # unpack the beta2 parameter
    alpha = []
    beta_major = []
    beta_minor = []
    for m, parameter in enumerate(parameters):
        # calculate the average width
        alpha_value = np.sqrt(parameter[2]*parameter[3])
        alpha.append(alpha_value)

        # check if beta1 and two got flipped
        # beta1 (beta major) should always be larger than beta2 (beta minor)
        if parameter[4] < parameter[5]:
            beta_major.append(parameter[5])
            beta_minor.append(parameter[4])
            # swap the error values
            sigma_beta_major[m], sigma_beta_minor[m] = sigma_beta_minor[m], sigma_beta_major[m]
        else:
            beta_major.append(parameter[4])
            beta_minor.append(parameter[5])
    # convert to numpy arrays
    alpha = np.asarray(alpha)
    beta_major = np.asarray(beta_major)
    beta_minor = np.asarray(beta_minor)

    # calculate the deviations for alpha
    sigma_alpha = .5*np.divide(np.sqrt(sigma_a**2 + sigma_b**2 + 2*ab_cov), alpha)


    # calculate the relative errors
    relative_alpha = sigma_alpha/alpha
    relative_flux = sigma_flux/calc_flux
    relative_beta_major = sigma_beta_major/beta_major
    relative_beta_minor = sigma_beta_minor/beta_minor

    # reject any parameters with relative errors above a certain amount
    mask = np.zeros(alpha.shape, dtype=bool)
    threshold = .49
    for m, tuple in enumerate(zip(relative_alpha, relative_flux, relative_beta_major, relative_beta_minor)):
        # if relative error is above the threshold, mask the corresponding parameter results
        if any(np.array(tuple) > threshold):
            mask[m] = True


    # calculate the beta ratios and deviation
    beta_ratio = np.divide(beta_major, beta_minor)
    beta_ratio_dev = np.sqrt(sigma_beta_major**2 + sigma_beta_minor**2 - 2*beta_cov)

    # apply the mask to the x values. this surpresses plotting of the value
    # measured_flux = np.ma.array(measured_flux, mask=mask)

    # plot the stuff
    plt.figure('alpha values')  # select correct figure
    plt.errorbar(measured_flux, alpha, yerr=sigma_alpha, ls='None', marker='o', capsize=3)
    plt.figure('beta values')  # select correct figure
    beta_major_plot.errorbar(measured_flux, beta_major, yerr=sigma_beta_major, ls='None', marker='o', capsize=3)
    beta_minor_plot.errorbar(measured_flux, beta_minor, yerr=sigma_beta_minor, ls='None', marker='o', capsize=3)
    plt.figure('beta ratio')
    # plt.errorbar(measured_flux, beta_ratio, yerr=beta_ratio_dev, ls='None', marker='o', capsize=3)
    plt.plot(measured_flux, beta_ratio, ls='None', marker='o')
    
plt.figure('alpha values')
plt.xlabel('Measured Flux (e-)')
plt.ylabel('Average Width alpha (pixels)')

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

plt.show()

