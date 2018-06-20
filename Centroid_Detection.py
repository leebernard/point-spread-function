# pass header data unit.  REMEBER, this is pass-by-reference
def bias_subtract(HDU):
    # import needed packages
    # import numpy as np
    # from astropy.io import fits
    import re
    from astropy.stats import sigma_clipped_stats

    # Store the data from the HDU argument
    Im_Data = HDU.data

    # pull the bias section information
    Bias_Sec = HDU.header['BIASSEC']
    print('Bias Section is ' + Bias_Sec)
    # print(type(Bias_Sec))
    # slice the string, for converting to int
    pattern = re.compile('\d+')  # pattern for all decimal digits
    print(pattern.findall(Bias_Sec))

    # hold the result in an object
    match = pattern.findall(Bias_Sec)

    # Bias section data from the header readout.
    # image is not indexed the same as python.
    # Image indexes (x,y), from lower left
    # python indexes (y,x)

    xmin = int(match[0])
    xmax = int(match[1])
    ymin = int(match[2])
    ymax = int(match[3])

    bias_data = Im_Data[ymin:ymax, xmin:xmax]

    # Calculate the bias, using clipped statistics in case of cosmic ray events, and print the 		#results
    bias_mean, bias_median, bias_std = sigma_clipped_stats(bias_data, sigma=3.0, iters=5)
    print('Bias mean: ' + str(bias_mean))
    print('Bias median: ' + str(bias_median))
    print('Bias standerd deviation: ' + str(bias_std))

    # calculate and print the bias area statistics, for reference.  DISABLED
    # print('Bias area after subtraction \n Mean: ')
    output_im = Im_Data - bias_mean
    return output_im


# calculates bias using a mask routine from photutils
def background_subtract(HDU):
    # import numpy as np
    # from astropy.io import fits

    # store the data from the HDU argument
    Im_Data = HDU.data

    # Generate mask
    from photutils import make_source_mask
    from astropy.stats import sigma_clipped_stats
    mask = make_source_mask(Im_Data, snr=2, npixels=5, dilate_size=11)

    # calculate bias using mean
    # clipped stats are used, just in case
    mean, median, std = sigma_clipped_stats(Im_Data, sigma=3.0, mask=mask)
    print('Background mean: ' + str(mean))
    print('Background median: ' + str(median))
    print('Background standerd deviation: ' + str(std))

    output_im = Im_Data - mean

    return output_im, mask


# needed packages
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# import re

from astropy.io import fits

# open fits file, best practice
file_name = '/home/lee/Documents/k4m_160319_101212_ori.fits.fz'
with fits.open(file_name) as hdu:
    hdu.info()
    data_im1 = hdu[1].data
    # bias subtraction
    bias_subtracted_im1 = bias_subtract(hdu[1])

# first object
# Centroid detection:
from photutils import centroid_com, centroid_1dg, centroid_2dg

# arbitrarily choosen object, section manually entered
ymin = 455
ymax = 505
xmin = 1490
xmax = 1540
Object1_Data = bias_subtracted_im1[ymin:ymax,xmin:xmax]

# Background subtract the object
Object1_Data, mask = background_subtract(Object1_Data)

# centroid techniques: need to learn the difference3s
# print('Centroids:')
# x1, y1 = centroid_com(Object1_Data)
# print((x1, y1))
# x2, y2 = centroid_1dg(Object1_Data)
# print((x2, y2))
# x3, y3 = centroid_2dg(Object1_Data)
# print((x3, y3))

# show an image of the aperture
from astropy.visualization import SqrtStretch

from astropy.visualization.mpl_normalize import ImageNormalize
norm = ImageNormalize(stretch=SqrtStretch())

# plt.figure()
f1, axisarg = plt.subplots(2,1)

axisarg[0].imshow(Object1_Data, norm=norm, origin='lower', cmap='viridis')
axisarg[1].imshow(mask, origin='lower', cmap='viridis')
# plt.show()




# Centroid dectection by curve fitting
# this attempt uses a bivariate normal distorbution as a model for the object

# define gaussian function, assuming no correlation between x and y
# indata is a pair of arrays, each array corresponding to the x indice or y indice, in the form (x, y)
# amplitude is the maximum amplitude of the function, minus background
# x0, y0 are the center coord. of the function
# sigma_x, sigma_y are the widths of the function
# offset is the background
# the output is flattened, in order to package it for curve_fit
def Gaussian_2d(indata, amplitude, x0, y0, sigma_x, sigma_y, offset):
    import numpy as np
    x, y = indata
    normalize = 1 / (sigma_x * sigma_y * 2 * np.pi)

    gaussian_fun = offset + amplitude * normalize * np.exp(
        -(x - x0) ** 2 / (2 * sigma_x ** 2) - (y - y0) ** 2 / (2 * sigma_y ** 2))

    return gaussian_fun.ravel()


# fit data to gaussian
from scipy.optimize import curve_fit
from photutils import centroid_2dg

# generate a best guess using photutils
x_guess, y_guess = centroid_2dg(Object1_Data)
amp_guess = np.amax(Object1_Data)

# indexes of the apature, remembering that python indexes vert, horz
y = np.arange(Object1_Data.shape[0])
x = np.arange(Object1_Data.shape[1])
x, y = np.meshgrid(x, y)

# curve fit
G_fit, G_cov = curve_fit(Gaussian_2d, (x, y), Object1_Data.ravel(), p0=[amp_guess, x_guess, y_guess, 1, 1, 1])
print('Resultant parameters')
print(G_fit)

error = np.sqrt(np.diag(G_cov))
print('Error on parameters')
print(error)

print('Covariance matrix, if that is interesting')
print(G_cov)

print('peak count')
print(np.amax(Object1_Data))

# display the result as a cross. The width of the lines correspond to the width in that direction
norm = ImageNormalize(stretch=SqrtStretch())

x_center = G_fit[1]
y_center = G_fit[2]
x_width = G_fit[3]
y_width = G_fit[4]

plt.figure()
plt.imshow(Object1_Data, norm=norm, origin='lower', cmap='viridis')
plt.errorbar(x_center, y_center, xerr=x_width, yerr=y_width, ecolor='red')


plt.show() # show all figures

print('center: ',x_center, ',', y_center)
print('width: ', 2*x_width, 'by', 2*y_width)

