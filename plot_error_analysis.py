

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

filename_list.append('/home/lee/Documents/decam-N9-A-archive.pkl')
filename_list.append('/home/lee/Documents/decam-N9-B-archive.pkl')
filename_list.append('/home/lee/Documents/decam-N4-A-archive.pkl')
filename_list.append('/home/lee/Documents/decam-N4-B-archive.pkl')
filename_list.append('/home/lee/Documents/decam-S5-A-archive.pkl')
filename_list.append('/home/lee/Documents/decam-S5-B-archive.pkl')
filename_list.append('/home/lee/Documents/decam-N5-A-archive.pkl')
filename_list.append('/home/lee/Documents/decam-N5-B-archive.pkl')
filename_list.append('/home/lee/Documents/decam-N3-A-archive.pkl')
filename_list.append('/home/lee/Documents/decam-N3-B-archive.pkl')

#
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

# list for storing relative error values
relative_err_store = []
plt.figure('Error vs sn ratio', figsize=(12, 10))
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
    plt.plot(ratio, relative_err[:,0], ls='None', marker='v', markersize=10, color='tab:blue')  # , label='Flux')
    plt.plot(ratio, relative_err[:,4], ls='None', marker='o', color='tab:red')  # , label='a')
    plt.plot(ratio, relative_err[:,5], ls='None', marker='+', markersize=12, color='tab:green')  # , label='b')
    plt.plot(ratio, relative_err[:,3], ls='None', marker='o', color='tab:purple')  # , label='beta')

# plt.yscale('log')
# plt.ylim(ymax=2)
plt.ylim(0, .25)
# plt.xlim(0, 530)
plt.title('Error vs SN DECam data MOSAIC-3')
plt.xlabel('Signal to Noise Ratio')
plt.ylabel('Relative Error of Fit Parameters (Linear Scale)')
plt.legend(('Flux', 'a', 'b', 'beta',), loc='best')


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
