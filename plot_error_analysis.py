

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
f1 = plt.figure('Error vs sn ratio')
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



    # unpack the measured flux
    measured_flux = []
    for aperture in apertures:
        measured_flux.append(np.sum(aperture))
    # convert to a numpy array. This is done separately, to avoid unnecessary copying of arrays
    measured_flux = np.asarray(measured_flux)

    # calculate signal to noise ratio
    noise = np.sqrt(measured_flux + aperture_size * background)

    # extract the relative error, defined as deviation/value
    relative_err = errors/parameters
    # add the relative error to the storage list, for the histogram
    relative_err_store.extend(relative_err[:, 0:6].flatten())

    # plot the ratio of flux to background vs error
    ratio = measured_flux/noise
    storage = []
    for m, ratio_holder in enumerate(ratio):
        storage.extend(np.ones(relative_err[m].size)*ratio_holder)

    plt.figure('Error vs sn ratio')
    flux = plt.plot(ratio, relative_err[:,0], ls='None', marker='o', color='tab:blue')  # , label='Flux')
    p = plt.plot(ratio, relative_err[:,1], ls='None', marker='o', color='tab:orange')  # , label='p')
    a = plt.plot(ratio, relative_err[:,2], ls='None', marker='+', markersize=12, color='tab:green')  # , label='a')
    b = plt.plot(ratio, relative_err[:,3], ls='None', marker='o', color='tab:red')  # , label='b')
    beta1 = plt.plot(ratio, relative_err[:,4], ls='None', marker='o', color='tab:purple')  # , label='beta1')
    beta2 = plt.plot(ratio, relative_err[:,5], ls='None', marker='o', color='tab:brown')  # , label='beta2')

# plt.yscale('log')
# plt.ylim(ymax=2)
plt.ylim(0, 1)
plt.title('Error vs Signal to Noise Ratio')
plt.xlabel('Signal to Noise Ratio')
plt.ylabel('Relative Error of Fit Parameters (Linear Scale)')
plt.legend(('Flux', 'p', 'a', 'b', 'beta1', 'beta2'), loc='best')


f2 = plt.figure('Error histogram')
hist_data = np.asarray(relative_err_store)
bins = np.linspace(0, 1, num=151)
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
