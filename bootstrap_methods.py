"""
A small module for writing bootstrap methods, such as re-sampling functions.
"""

import numpy as np

"""
Take a paired set of data, and resample the set
"""

iters = 10000
row_sample_fits = []
col_sample_fits = []
for _ in range(iters):

    # generate an array of integers that correspond to the size of the data set
    indexarray = np.arange(len(flux_values))

    # sample the array locations
    # if replace is True, points in parent population can be reused in sample
    sample_indexs = np.random.choice(indexarray, size=len(indexarray), replace=True)

    # generate fit

    row_sample_fits.append(np.ma.polyfit(flux_values[sample_indexs], row_delta_list[sample_indexs], deg=1))
    col_sample_fits.append(np.ma.polyfit(flux_values[sample_indexs], col_delta_list[sample_indexs], deg=1))

row_sample_fits = np.asarray(row_sample_fits)
col_sample_fits = np.asarray(col_sample_fits)

print(np.mean(row_sample_fits, axis=0))
print(np.std(row_sample_fits, axis=0))