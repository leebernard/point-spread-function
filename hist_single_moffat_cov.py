

import pickle
import numpy as np
import matplotlib.pyplot as plt

"""
This is a script for plotting the results from fitting astronomical objects

The fit results are from fitting astronomical objects to a sum of two 
moffat functions. The data (aperture) that was used to produce each fit is 
included, as well as the location of the lower left corner of each aperture

Source function:
flat_elliptical_Moffat(indata, flux, x0, y0, beta, a, b, theta):
"""
def cov_to_coeff(cov):
    """
    This function converts a covarience matrix to a correlation coefficient matrix

    It does so by normalizing each entry in the matrix by the product of the
    of the relevant variances"""
    # make a copy of the covarience array
    coeff = np.copy(cov)

    it = np.nditer(coeff, flags=['multi_index'], op_flags=['writeonly'])
    while not it.finished:
        it[0] = it[0]/(np.sqrt(cov[it.multi_index[0], it.multi_index[0]]) * np.sqrt(cov[it.multi_index[1], it.multi_index[1]]))
        it.iternext()

    return coeff
filename_list = []
# filename = '/home/lee/Documents/single-moffat-archive-im7.pkl'
filename_list.append('/home/lee/Documents/single-moffat-archive-im1.pkl')
filename_list.append('/home/lee/Documents/single-moffat-archive-im2.pkl')
filename_list.append('/home/lee/Documents/single-moffat-archive-im4.pkl')
filename_list.append('/home/lee/Documents/single-moffat-archive-im5.pkl')
filename_list.append('/home/lee/Documents/single-moffat-archive-im7.pkl')

filename_list.append('/home/lee/Documents/single-moffat-archive-im9.pkl')
filename_list.append('/home/lee/Documents/single-moffat-archive-im10.pkl')
filename_list.append('/home/lee/Documents/single-moffat-archive-im12.pkl')
filename_list.append('/home/lee/Documents/single-moffat-archive-im13.pkl')
filename_list.append('/home/lee/Documents/single-moffat-archive-im16.pkl')



fluxbeta_coeff = []
fluxa_coeff = []
fluxb_coeff = []
ab_coeff = []
abeta_coeff = []
bbeta_coeff = []
for n, filename in enumerate(filename_list):
    with open(filename, mode='rb') as file:
        archive = pickle.load(file)

    cov = archive['param_cov']


    # list to hold the deviations on the parameters
    error_list = []

    for cov_mat in cov:
        error_list.append(np.sqrt(np.diag(cov_mat)))

        # convert covariance matrix to correlation coeff
        coeff_mat = cov_to_coeff(cov_mat)
        # unpack the correlation coeff between the flux and other parameters
        fluxbeta_coeff.append(coeff_mat[0][3])
        fluxa_coeff.append(coeff_mat[0][4])
        fluxb_coeff.append(coeff_mat[0][5])

        # unpack the correlation coeff between the a and b parameters
        ab_coeff.append(coeff_mat[4][5])

        # unpack the correlation coeff between the a, b and beta parameters
        abeta_coeff.append(coeff_mat[4][3])
        bbeta_coeff.append(coeff_mat[5][3])

# convert to numpy arrays
fluxbeta_coeff = np.asarray(fluxbeta_coeff)
fluxa_coeff = np.asarray(fluxa_coeff)
fluxb_coeff = np.asarray(fluxb_coeff)
ab_coeff = np.asarray(ab_coeff)
abeta_coeff = np.asarray(abeta_coeff)
bbeta_coeff = np.asarray(bbeta_coeff)


# # figures for plotting
# flux_hist = plt.figure('Flux coeff', figsize=(12,10))
# flux_a_coeff = flux_hist.add_subplot(311)
# flux_b_coeff = flux_hist.add_subplot(312)
# flux_beta_coeff = flux_hist.add_subplot(313)
bins = np.linspace(0, 1, num=251)

# flux_a_coeff.hist(np.clip(fluxa_coeff, bins[0], bins[-1]), bins=bins, color='tab:blue')
plt.figure('flux a coeff')
plt.hist(fluxa_coeff, bins=bins, color='tab:blue')
plt.figure('flux b coeff')
plt.hist(fluxb_coeff, bins=bins, color='tab:blue')
plt.figure('flux beta')
plt.hist(fluxbeta_coeff, bins=bins, color='tab:blue')


plt.show()
