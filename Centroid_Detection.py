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

plt.figure()
plt.imshow(Object1_Data, norm=norm, origin='lower', cmap='viridis')
plt.show()


