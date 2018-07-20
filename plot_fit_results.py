

import pickle
import numpy as np
import matplotlib.pyplot as plt

"""
This is a script for plotting the results from fitting astronomical objects

The fit results are from fitting astronomical objects to a sum of two 
moffat functions. The data (aperture) that was used to produce each fit is 
included, as well as the location of the lower left corner of each aperture

"""

# load the data from Im1
filename = '/home/lee/Documents/sample-archive-im16.pkl'

with open(filename, mode='rb') as file:
    archive_im1 = pickle.load(file)

apertures_im1 = archive_im1['apertures']
parameters_im1 = archive_im1['parameters']
cov_im1 = archive_im1['param_cov']

# list to hold the deviations on the parameters
error_im1 = []
for cov_mat in cov_im1:
    error_im1.append(np.sqrt(np.diag(cov_mat)))
# convert to numpy array, for convience
error_im1 = np.asarray(error_im1)

# unpack the measured flux
measured_flux_im1 = []
for aperture in apertures_im1:
    measured_flux_im1.append(np.sum(aperture))
# convert to a numpy array. This is done later, to avoid unnecessary copying of arrays
measured_flux_im1 = np.asarray(measured_flux_im1)

# unpack the calculated flux
calc_flux = []
for parameter in parameters_im1:
    calc_flux.append(parameter[0])
# convert to numpy array
calc_flux = np.asarray(calc_flux)

flux_ratio = calc_flux/measured_flux_im1

flux_ratio_dev = error_im1[:, 0]/measured_flux_im1

f1 = plt.figure()
plt.errorbar(measured_flux_im1, flux_ratio, yerr=flux_ratio_dev, ls='None')


