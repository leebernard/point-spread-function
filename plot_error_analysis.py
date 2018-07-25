

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

# list for storing relative error values
relative_err_store = []
f1 = plt.figure('Error vs flux to background ratio')
for n, filename in enumerate(filename_list):
    with open(filename, mode='rb') as file:
        archive = pickle.load(file)


    apertures = archive['apertures']
    background = archive['background']
    parameters = archive['parameters']
    cov = archive['param_cov']

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

    background_flux = aperture_size*background

    # unpack the measured flux
    measured_flux = []
    for aperture in apertures:
        measured_flux.append(np.sum(aperture))
    # convert to a numpy array. This is done separately, to avoid unnecessary copying of arrays
    measured_flux = np.asarray(measured_flux)

    # extract the relative error, defined as deviation/value
    relative_err = errors/parameters
    # add the relative error to the storage list, for the histogram
    relative_err_store.extend(relative_err.flatten())

    # plot the ratio of flux to background vs error
    ratio = measured_flux/background_flux
    storage = []
    for m, ratio_holder in enumerate(ratio):
        storage.extend(np.ones(relative_err[m].size)*ratio_holder)

    plt.figure('Error vs flux to background ratio')
    flux = plt.plot(relative_err[:,0], ratio, ls='None', marker='o', color='tab:blue', label='Flux')
    p = plt.plot(relative_err[:,1], ratio, ls='None', marker='o', color='tab:orange', label='p')
    a = plt.plot(relative_err[:,2], ratio, ls='None', marker='o', color='tab:green', label='a')
    b = plt.plot(relative_err[:,3], ratio, ls='None', marker='o', color='tab:red', label='b')
    beta1 = plt.plot(relative_err[:,4], ratio, ls='None', marker='o', color='tab:purple', label='beta1')
    beta2 = plt.plot(relative_err[:,5], ratio, ls='None', marker='o', color='tab:brown', label='beta2')


plt.xlim(0, 1)

f2 = plt.figure('Error histogram')
hist_data = np.asarray(relative_err_store)
bins = np.linspace(0, 1, num=100)
# bins = np.append(bins, hist_data.max())
plt.hist(hist_data, bins=bins)
plt.show()

