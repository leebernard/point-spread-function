

import pickle
import numpy as np
import matplotlib.pyplot as plt

"""
This is a script for plotting the results from fitting astronomical objects

The fit results are from fitting astronomical objects to a sum of two 
moffat functions. The data (aperture) that was used to produce each fit is 
included, as well as the location of the lower left corner of each aperture

This particular result is for checking the ratio in a sum of two moffat 
functions model of the PSF
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
ratio_figure = plt.figure('flux ratio values', figsize=(12, 10))


for n, filename in enumerate(filename_list):
    with open(filename, mode='rb') as file:
        archive = pickle.load(file)


    apertures = archive['apertures']
    parameters = archive['parameters']
    cov = archive['param_cov']

    # list to hold the deviations on the parameters
    errorlist = []
    for cov_mat in cov:
        errorlist.append(np.sqrt(np.diag(cov_mat)))
    # convert to numpy array, for convenience
    errorlist = np.asarray(errorlist)
    # unpack the needed errors
    p_dev = errorlist[:, 1]

    # unpack the measured flux
    measured_flux = []
    for aperture in apertures:
        measured_flux.append(np.sum(aperture))
    # convert to a numpy array. This is done later, to avoid unnecessary copying of arrays
    measured_flux = np.asarray(measured_flux)

    # unpack the p ratio
    p_ratio = []
    for parameter in parameters:
        current_p = parameter[1]
        if current_p < .5:
            current_p = 1-current_p
        p_ratio.append(current_p)
    # convert to numpy array
    p_ratio = np.asarray(p_ratio)

    # mask the results with huge errors
    # p_dev = np.ma.masked_greater(p_dev, 1)
    # measured_flux = np.ma.masked_array(measured_flux, mask=p_dev.mask)
    # p_ratio = np.ma.masked_array(p_ratio, mask=p_dev.mask)

    labelstr = f'Frame {n}'
    plt.errorbar(measured_flux, p_ratio, yerr=p_dev, ls='None', marker='o', capsize=2, label=labelstr)

plt.xlabel('Measured Flux (e-)')
plt.title('Ratio by which flux is divided between the two PSF\'s')
plt.legend(loc='best')
plt.xlim(xmax=850000)
plt.ylim(0, 1.5)
plt.show()

