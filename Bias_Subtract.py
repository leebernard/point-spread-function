# pass header data unit.  REMEBER, this is pass-by-reference
def bias_subtract(HDU):
    # import needed packages
    import numpy as np
    from astropy.io import fits
    import re
    from astropy.stats import sigma_clipped_stats
    
    # Store the data from the HDU argument
    Im_Data = HDU.data
    
    
    # pull the bias section information
    Bias_Sec = HDU.header['BIASSEC']
    print(Bias_Sec)
    print(type(Bias_Sec))
    # slice the string, for converting to int
    pattern = re.compile('\d+') # pattern for all decimal digits
    print(pattern.findall(Bias_Sec))

    # hold the result in an object
    match = pattern.findall(Bias_Sec)
    
    
    
    # Bias section data from the header readout.  
    bias_data = Im_Data[int(match[0]):int(match[1]),int(match[2]):int(match[3])]
    
    # Calculate the bias, using clipped statistics in case of cosmic ray events, and print the 		#results
    bias_mean, bias_median, bias_std = sigma_clipped_stats(bias_data, sigma=3.0, iters=5)
    print('Bias mean: '+ str(bias_mean))
    print('Bias median: '+str(bias_median))
    print('Bias standerd deviation: '+str(bias_std))
    
    # calculate and print the bias area statistics, for reference.  DISABLED
    # print('Bias area after subtraction \n Mean: ')
    output_im = Im_Data-bias_mean
    return output_im
