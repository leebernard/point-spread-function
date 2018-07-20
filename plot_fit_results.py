

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
filenames = []

filenames.append('/home/lee/Documents/sample-archive-im1.pkl')
filenames.append('/home/lee/Documents/sample-archive-im2.pkl')
filenames.append('/home/lee/Documents/sample-archive-im4.pkl')
filenames.append('/home/lee/Documents/sample-archive-im5.pkl')
filenames.append('/home/lee/Documents/sample-archive-im7.pkl')

# filenames.append('/home/lee/Documents/sample-archive-im9.pkl')
# filenames.append('/home/lee/Documents/sample-archive-im10.pkl')
# filenames.append('/home/lee/Documents/sample-archive-im12.pkl')
# filenames.append('/home/lee/Documents/sample-archive-im13.pkl')
# filenames.append('/home/lee/Documents/sample-archive-im16.pkl')

# figure for ploting the flux ratios
f1 = plt.figure()
# n = 0
for filename in filenames:
    with open(filename, mode='rb') as file:
        archive = pickle.load(file)

    # iterate counter
    n += 1

    apertures = archive['apertures']
    parameters = archive['parameters']
    cov = archive['param_cov']
    
    # list to hold the deviations on the parameters
    error = []
    for cov_mat in cov:
        error.append(np.sqrt(np.diag(cov_mat)))
    # convert to numpy array, for convience
    error = np.asarray(error)
    
    # unpack the measured flux
    measured_flux = []
    for aperture in apertures:
        measured_flux.append(np.sum(aperture))
    # convert to a numpy array. This is done later, to avoid unnecessary copying of arrays
    measured_flux = np.asarray(measured_flux)
    
    # unpack the calculated flux
    calc_flux = []
    for parameter in parameters:
        calc_flux.append(parameter[0])
    # convert to numpy array
    calc_flux = np.asarray(calc_flux)
    
    flux_ratio = calc_flux/measured_flux
    
    flux_ratio_dev = error[:, 0]/measured_flux

    labelstr = f'Frame {n}'
    plt.errorbar(measured_flux, flux_ratio, yerr=flux_ratio_dev, ls='None', marker='o', capsize=2, label=labelstr)

plt.xlabel('Measured Flux (e-)')
plt.ylabel('Ratio of Flux parameter to Measured Flux')
plt.legend(loc='best')
plt.xlim(xmax=850000)
plt.ylim(.90, 1.05)
plt.show()
