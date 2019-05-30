

import pickle
import numpy as np
import matplotlib.pyplot as plt

"""
This is a script for plotting the results from fitting the PSF of
astronomical point source objects

The fit results are from fitting astronomical objects to a moffat 
distribution. The data (aperture) that was used to produce each fit is 
included, as well as the location of the lower left corner of each aperture

The results are analysis of the widths, as well as eccentrities and other 
stuff derived from the fit parameters.

flat_elliptical_Moffat(indata, flux, x0, y0, beta, a, b, theta):
"""
# filename_listA = []
# filename_listB = []
# # convient holder for legend
# legend_list = []
# filename_listA.append('/home/lee/Documents/decman-fit-archive-20170331/decam-94s-S4-A-archive.pkl')
# legend_list.append('S4 2:48UT-94s')
# filename_listB.append('/home/lee/Documents/decman-fit-archive-20170331/decam-94s-S4-B-archive.pkl')
# # legend_list.append('94s-S4-B')
# filename_listA.append('/home/lee/Documents/decman-fit-archive-20170331/decam-94s-N4-A-archive.pkl')
# legend_list.append('N4 2:48UT-94s')
# filename_listB.append('/home/lee/Documents/decman-fit-archive-20170331/decam-94s-N4-B-archive.pkl')
# # legend_list.append('94s-N4')
#
# filename_listA.append('/home/lee/Documents/decman-fit-archive-20170331/decam-91s-S4-A-archive.pkl')
# legend_list.append('S4 2:16UT-91s')
# filename_listB.append('/home/lee/Documents/decman-fit-archive-20170331/decam-91s-S4-B-archive.pkl')
# # legend_list.append('91s-S4-B')
# filename_listA.append('/home/lee/Documents/decman-fit-archive-20170331/decam-91s-N4-A-archive.pkl')
# legend_list.append('N4 2:16UT-91s')
# filename_listB.append('/home/lee/Documents/decman-fit-archive-20170331/decam-91s-N4-B-archive.pkl')
# # legend_list.append('91s-N4-B')
#
# filename_listA.append('/home/lee/Documents/decman-fit-archive-20170331/decam-102s-S4-A-archive.pkl')
# legend_list.append('S4 2:11UT-102s')
# filename_listB.append('/home/lee/Documents/decman-fit-archive-20170331/decam-102s-S4-B-archive.pkl')
# # legend_list.append('102s-S4-B')
# filename_listA.append('/home/lee/Documents/decman-fit-archive-20170331/decam-102s-N4-A-archive.pkl')
# legend_list.append('N4 2:11UT-102s')
# filename_listB.append('/home/lee/Documents/decman-fit-archive-20170331/decam-102s-N4-B-archive.pkl')
# legend_list.append('102s-N4-B')

# # filename_list.append('/home/lee/Documents/decam-N9-A-archive.pkl')
# # legend_list.append('CCD N9, amp A')
# filename_list.append('/home/lee/Documents/decam-N9-B-archive.pkl')
# legend_list.append('CCD N9, amp B')
# filename_list.append('/home/lee/Documents/decam-N4-A-archive.pkl')
# legend_list.append('CCD N4, amp A')
# filename_list.append('/home/lee/Documents/decam-N4-B-archive.pkl')
# legend_list.append('CCD N4, amp B')
# filename_list.append('/home/lee/Documents/decam-S5-A-archive.pkl')
# legend_list.append('CCD S5, amp A')
# filename_list.append('/home/lee/Documents/decam-S5-B-archive.pkl')
# legend_list.append('CCD S5, amp B')
# filename_list.append('/home/lee/Documents/decam-N5-A-archive.pkl')
# legend_list.append('CCD N5, amp A')
# # filename_list.append('/home/lee/Documents/decam-N5-B-archive.pkl')
# # legend_list.append('CCD N5, amp B')
# filename_list.append('/home/lee/Documents/decam-N3-A-archive.pkl')
# legend_list.append('CCD N3, amp A')
# filename_list.append('/home/lee/Documents/decam-N3-B-archive.pkl')
# legend_list.append('CCD N3, amp B')
# filename_list.append('/home/lee/Documents/decam-N16-A-archive.pkl')
# legend_list.append('CCD N16, amp A')
# filename_list.append('/home/lee/Documents/decam-N16-B-archive.pkl')
# legend_list.append('CCD N16, amp B')

# ('CCD N9, amp A', 'CCD N9, amp B', ' CCD N4, amp A', 'CCD N4, amp B', 'CCD S5, amp A', 'CCD S5, amp B', 'CCD N5, amp A','CCD N5, amp B')

# filename = '/home/lee/Documents/single-moffat-archive-im7.pkl'

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
# filename_list.append('/home/lee/Documents/single-moffat-archive-im16.pkl')

# filename = '/home/lee/Documents/decam-ccds-N4-S4-20170331-unbiased-archive.pkl'
filename = '/home/lee/Documents/decam-ccds-N4-S4-20170331-unbiased-forcedangle-archive.pkl'
# open the archive
with open(filename, mode='rb') as file:
    archive_list = pickle.load(file)

# split the archived data, so that it can be paired up
archive_listA = archive_list[::2]
archive_listB = archive_list[1::2]
# figures for plotting
plt.figure('alpha values', figsize=(12, 10))
plt.figure('beta values', figsize=(12, 10))
plt.figure('FWHM', figsize=(12, 10))
# plt.figure('angle vs peak pixel value',figsize=(12, 10))
plt.figure('Eccentricity vs peak pixel value', figsize=(12, 10))

'''
for n, filenameA in enumerate(filename_listA):
    with open(filenameA, mode='rb') as file:
        archive = pickle.load(file)

    apertures = archive['apertures']
    parameters = archive['parameters']
    background = archive['background']
    cov = archive['param_cov']


    with open(filename_listB[n], mode='rb') as file:
        archive = pickle.load(file)
    apertures.extend(archive['apertures'])
    parameters.extend(archive['parameters'])
    background.extend(archive['background'])
    cov.extend(archive['param_cov'])
'''
legend_list = []
angle_values_list = []

# retrieve default colors
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = iter(prop_cycle.by_key()['color'])

for n, archive in enumerate(archive_listA):
    # with open(filename, mode='rb') as file:
    #     archive = pickle.load(file)
    apertures = archive['apertures']
    background = archive['background']
    parameters = archive['parameters']
    cov = archive['param_cov']
    legend_list.append(archive['dataset_name'])
    # add the second half of the CCD. Extend extends the current entries, rather than adding new ones.
    # with open(filename_listB[n], mode='rb') as file:
    #     archive = pickle.load(file)
    archiveB = archive_listB[n]
    apertures.extend(archiveB['apertures'])
    parameters.extend(archiveB['parameters'])
    background.extend(archiveB['background'])
    cov.extend(archiveB['param_cov'])
    parameters = np.asarray(parameters)
    background = np.asarray(background)

    # list to hold the deviations on the parameters
    error_list = []
    ab_cov = []
    abeta_cov = []
    bbeta_cov = []
    # # 95% confidence interval for 7 parameters
    # delta_chisqrd = 14.1
    # 95% confidence interval for 6 parameters
    delta_chisqrd = 12.8
    for cov_mat in cov:
        error_list.append(np.sqrt(np.diag(cov_mat)*delta_chisqrd))

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
    sigma_theta = error_list[:, -1]


    # unpack the measured Flux
    measured_flux = []
    max_pixel = []
    for aperture in apertures:
        measured_flux.append(np.sum(aperture))
        max_pixel.append(np.max(aperture))
    # convert to a numpy array. This is done separately, to avoid unnecessary copying of arrays
    measured_flux = np.asarray(measured_flux)
    max_pixel = np.asarray(max_pixel)

    # extract the aperture size
    # aperture_size = [apt.size for apt in apertures]
    aperture_size = []
    for apt in apertures:
        aperture_size.append(apt.size)
    # convert to array for convience
    aperture_size = np.asarray(aperture_size)

    # split background apart into values and deviation
    background, bkg_dev = background[:, 0], background[:, 1]

    # calculate signal to noise ratio
    noise = np.sqrt(measured_flux + aperture_size * background)

    sn_ratio = measured_flux/noise





    # unpack the width parameters a and b
    # unpack the beta parameter
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


    """
    create a boolean mask that clips values from the plot that have a s/n less than 60
    """
    # clip any values that have a S/N ration below the threshold
    sn_clip_mask = np.zeros(measured_flux.size, dtype=bool)
    threshold = 100
    for n, _ in enumerate(measured_flux):
        # if relative error is above the threshold, mask the corresponding parameter results
        if sn_ratio[n] < threshold or measured_flux[n] > 5000000:
            print('Got one!', measured_flux[n])
            sn_clip_mask[n] = True
    # apply the mask to the x values. this surpresses plotting of the value
    max_pixel = np.ma.array(max_pixel, mask=sn_clip_mask)
    measured_flux = np.ma.array(measured_flux, mask=sn_clip_mask)
    # # apply the mask to the angle values, and then store them for histograming
    # angle_values = (np.ma.array(parameters[:, -1], mask=sn_clip_mask))
    # angle_values_list.extend(angle_values)

    # calculate the average width, alpha
    alpha = np.sqrt(a_param * b_param)
    # calculate the deviations for alpha
    sigma_alpha = .5 * np.sqrt(b_param / a_param * (sigma_a ** 2) + a_param / b_param * (sigma_b ** 2) + 2 * ab_cov)
    # make a linear fit
    poly_coeffs, poly_cov = np.ma.polyfit(measured_flux, alpha, deg=1, w=1/sigma_alpha, cov=True)
    print(poly_coeffs)
    # make sets of data to show the fit
    x_values = np.arange(measured_flux.min(), measured_flux.max())
    y_values = poly_coeffs[0]*x_values + poly_coeffs[1]

    # calculate the Full Width, Half Max
    fwhm = 2*alpha
    sigma_fwhm = 2*sigma_alpha

    # plot the stuff
    color = next(colors)
    plt.figure('alpha values')  # select correct figure
    plt.errorbar(measured_flux, alpha, yerr=sigma_alpha, ls='None', marker='o', capsize=3, color=color)
    plt.plot(x_values, y_values, color=color)

    # plt.figure('beta values')  # select correct figure
    # plt.errorbar(measured_flux, beta, yerr=sigma_beta, ls='None', marker='o', capsize=3)
    #
    # plt.figure('FWHM')
    # plt.errorbar(measured_flux, fwhm, yerr=sigma_fwhm, ls='None', marker='o', capsize=3)
    #
    # # plt.figure('angle vs peak pixel value')
    # # plt.errorbar(measured_flux, angle_values * 57.2958, yerr=sigma_theta * 57.2958, ls='None', marker='o', capsize=3)
    #
    # plt.figure('Eccentricity vs peak pixel value')
    # plt.errorbar(measured_flux, (a_param-b_param)/alpha, yerr=sigma_alpha/alpha, ls='None', marker='o', capsize=3)
plt.figure('alpha values')
plt.title('Single Moffat Alpha Values(Half Width Half Max)')
plt.xlabel('Measured Flux of Object (e-)')
plt.ylabel('Average Width alpha (pixels)')
# plt.ylim(1.5, 2.5)
# plt.legend(('Frame 1', 'Frame 2', 'Frame 4', 'Frame 5', 'Frame 7', 'Frame 9', 'Frame 10', 'Frame 12', 'Frame 13', 'Frame 16'), loc='best')
plt.legend(legend_list)

plt.figure('beta values')
plt.title('Single Moffat Beta Values')
plt.ylabel(r'$\beta$ Value')
plt.xlabel('Measured Flux of Object (e-)')
# plt.ylim(3.5, 5.5)
plt.legend(legend_list)

plt.figure('FWHM')
plt.title('17/03/31 r filter RA:164-167arcmin DEC:2.5-4.5arcmin WCS')
plt.xlabel('Measured Flux of Object (e-)')
plt.ylabel('Full Width, Half Maximum (pixels)')
# plt.ylim(3, 4)
plt.legend(legend_list)

# plt.figure('angle vs peak pixel value')
# plt.title('Angle vs Peak pixel Value')
# plt.xlabel('Measured Flux of Object (e-)')
# plt.ylabel('Angle of Eccentricity (degrees)')
# plt.ylim(-10, 90)

plt.figure('Eccentricity vs peak pixel value')
plt.title('Eccentricity vs Peak Pixel Value')
plt.xlabel('Measured Flux of Object (e-)')
plt.ylabel('Percent Eccentricity (a-b)/HWHM')
# plt.ylim(-.1, .2)

# # commented out the histogram, cause it takes too long
# plt.figure('angle histogram')
# plt.title('Histogram of eccentricity angles')
# bins = np.linspace(0, 90, num=91)
# plt.hist(np.asarray(angle_values_list) * 57.2958, bins=bins)  # angles converted from radians to degrees

plt.show()