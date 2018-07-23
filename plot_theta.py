

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
f1 = plt.figure('Ratio of Measured to Calculated Flux')

for n, filename in enumerate(filename_list):
    with open(filename, mode='rb') as file:
        archive = pickle.load(file)


    apertures = archive['apertures']
    parameters = archive['parameters']
    cov = archive['param_cov']

    # list to hold the deviations on the parameters
    error_list = []
    for cov_mat in cov:
        error_list.append(np.sqrt(np.diag(cov_mat)))
    # convert to numpy array, for convenience
    error_list = np.asarray(error_list)
    # unpack the needed error
    theta_dev = error_list[:, -1]

    # unpack the theta values
    theta = []
    for parameter in parameters:
        theta.append(parameter[-1])
    # convert to numpy array
    theta = np.asarray(theta)

    # unpack the measured flux
    measured_flux = []
    for aperture in apertures:
        measured_flux.append(np.sum(aperture))
    # convert to a numpy array. This is done later, to avoid unnecessary copying of arrays
    measured_flux = np.asarray(measured_flux)

    frame_number = n+1
    labelstr = f'Frame {frame_number}'

    frame = np.ones(theta.size) * (n+1)
    plt.errorbar(measured_flux, theta , yerr=theta_dev, ls='None', marker='o', capsize=2, label=labelstr)



