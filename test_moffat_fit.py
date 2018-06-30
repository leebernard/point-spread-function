"""This file is for testing fit models, by generating fake data

Expect this file to be somewhat messy. Also, expect everything in this file to be unexpectedly deleted"""
# needed modules
import numpy as np
import matplotlib.pyplot as plt

# needed functions
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize


# functions to be tested
def Moffat(indata, flux, x0, y0, alpha, beta, offset):
    """Model of PSF using a single Moffat distribution
    """
    x, y = indata
    normalize = (beta - 1) / (np.pi * alpha ** 2)

    moffat_fun = offset + flux*normalize*(1 + ((x - x0)**2 + (y - y0)**2) / (alpha**2))**(-beta)

    return moffat_fun


def moffat_fit(indata):
    """wrapper for the moffat fit procedure.

    This fit is rather complicated, so it has been wrapped into a function for convience
    """

    def flat_Moffat(indata, flux, x0, y0, alpha, beta, offset):
        """Model of PSF using a single Moffat distribution

        This function flattens the output, for curve fitting
        """
        x, y = indata
        normalize = (beta-1)/(np.pi*alpha**2)

        moffat_fun = offset + flux*normalize*(1 + ((x-x0)**2 + (y-y0)**2)/(alpha**2))**(-beta)

        return moffat_fun.ravel()


    # instead fit data to moffat
    from scipy.optimize import curve_fit

    # indexes of the aperture, remembering that python indexes vert, horz
    y = np.arange(indata.shape[0])
    x = np.arange(indata.shape[1])
    x, y = np.meshgrid(x, y)

    # generate a best guess
    flux_guess = np.amax(indata)
    y_guess = indata.shape[0]/2
    x_guess = indata.shape[1]/2
    alpha_guess = 4
    beta_guess = 2
    offset_guess = 0

    guess = [flux_guess, x_guess, y_guess, alpha_guess, beta_guess, offset_guess]

    # generate parameters for fit
    m_fit, m_cov = curve_fit(flat_Moffat, (x, y), indata.ravel(), p0=guess)


    """Chi squared calculations
    """
    observed = indata.ravel()

    m_input = (x, y)
    flux = m_fit[0]
    x0 = m_fit[1]
    y0 = m_fit[2]
    alpha = m_fit[3]
    beta = m_fit[4]
    offset = m_fit[5]

    expected = flat_Moffat(m_input, flux, x0, y0, alpha, beta, offset)

    # calculated raw chi squared
    chisq = sum(np.divide((observed - expected) ** 2, expected))

    # degrees of freedom, 5 parameters
    degrees_of_freedom = observed.size - 6

    # normalized chi squared
    chisq_norm = chisq / degrees_of_freedom

    print('normalized chi squared:')
    print(chisq_norm)
    return m_fit, m_cov


# generate the fake object
y = np.arange(50)
x = np.arange(50)
x, y = np.meshgrid(x, y)
m_input = (x, y)
flux = 1000000  # 1 million
x0 = 26
y0 = 23
alpha = 6
beta = 9
offset = 0
fake_object = Moffat(m_input, flux, x0, y0, alpha, beta, offset)

# spike the object with some noise
noise = np.random.normal(0,25,fake_object.shape)
fake_object = fake_object + noise

# show the generated object
norm = ImageNormalize(stretch=SqrtStretch())

plt.figure()
plt.imshow(fake_object, norm=norm, origin='lower', cmap='viridis')

# fit the fake data
m_fit, m_cov = moffat_fit(fake_object)

print('Resultant parameters')
print('Flux: ' + str(m_fit[0]))
print('Center (x, y): '+str(m_fit[1]) + ', ' + str(m_fit[2]))
print('alpha: '+str(m_fit[3]))
print('beta: ' + str(m_fit[4]))
print('background: ' + str(m_fit[5]))

error = np.sqrt(np.diag(m_cov))
print('Error on parameters')
print(error)


plt.show()