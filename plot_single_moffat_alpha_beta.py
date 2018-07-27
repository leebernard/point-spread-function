

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

# filename = '/home/lee/Documents/single-moffat-archive-im7.pkl'
filename = '/home/lee/Documents/single-moffat-archive-im7-without-hf.pkl'

with open(filename, mode='rb') as file:
    archive = pickle.load(file)

apertures = archive['apertures']
parameters = archive['parameters']
cov = archive['param_cov']

# list to hold the deviations on the parameters
error_list = []
ab_cov = []
abeta_cov = []
bbeta_cov = []
beta_cov = []
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

# calculate the relative errors
relative_alpha = sigma_alpha / alpha
relative_flux = sigma_flux / calc_flux
relative_beta = sigma_beta / beta

# calculate the Full Width, Half Max
fwhm = 2*alpha*np.sqrt(2**(1/beta) - 1)
varience_fwhm = ((np.sqrt(2**(1/beta) - 1))*sigma_alpha)**2 \
             + (np.divide(alpha*2**(1/beta)*np.log(1/2), (2*np.sqrt(2**(1/beta) - 1)*beta**2))*sigma_beta)**2 \
             + (np.sqrt(b_param/a_param)*abeta_cov + np.sqrt(a_param/b_param)*bbeta_cov) * (alpha*np.log(1/2)*2**(1/beta))/(2*beta**2)
sigma_fwhm = 2*np.sqrt(varience_fwhm)

# reject any parameters with relative errors above a certain amount
mask = np.zeros(alpha.shape, dtype=bool)
threshold = 1
for m, tuple in enumerate(zip(relative_alpha, relative_flux, relative_beta)):
    # if relative error is above the threshold, mask the corresponding parameter results
    if any(np.array(tuple) > threshold):
        mask[m] = True
# apply the mask to the x values. this surpresses plotting of the value
measured_flux = np.ma.array(measured_flux, mask=mask)

# plot the stuff
plt.figure('alpha values', figsize=(7, 6))  # select correct figure
plt.errorbar(measured_flux, alpha, yerr=sigma_alpha, ls='None', marker='o', capsize=3)

plt.figure('beta values', figsize=(7, 6))  # select correct figure
plt.errorbar(measured_flux, beta, yerr=sigma_beta, ls='None', marker='o', capsize=3)

plt.figure('Full Width, Half Maximum', figsize=(7, 6))
plt.errorbar(measured_flux, fwhm, yerr=sigma_fwhm, ls='None', marker='o', capsize=3)

plt.figure('alpha values')
plt.title('Single Moffat Alpha Values')
plt.xlabel('Measured Flux (e-)')
plt.ylabel('Average Width alpha (pixels)')
plt.ylim(3, 6)

plt.figure('beta values')
plt.title('Single Moffat Beta Values')
plt.ylabel(r'$\beta$ Value')
plt.xlabel('Measured Flux (e-)')

plt.figure('Full Width, Half Maximum')
plt.title('Test version')
plt.xlabel('Measured Flux (e-)')
plt.ylabel('Full Width, Half Maximum (pixels)')
plt.ylim(3, 5)

plt.show()