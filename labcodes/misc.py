"""Module contains miscellaneous useful functions.
"""

import numpy as np
from pathlib import Path
from scipy.optimize import fsolve

def auto_rotate(data, with_angle=False):
    angle = -0.5 * np.angle(np.mean(data**2) - np.mean(data)**2)  # Minize imag(var(data)), by Kaho
    
    if with_angle is True:
        return data * np.exp(1j*angle), angle  # counter-clockwise
    else:
        return data * np.exp(1j*angle)

def remove_e_delay(angle, freq):
    """Returns phase without linear freq dependence."""
    angle = np.unwrap(angle)
    e_delay = (angle[-1] - angle[0]) / (freq[-1] - freq[0])
    angle -= e_delay * (freq - freq[0])
    angle -= angle[0]
    return angle

def find_freq_guess(x, y):
    """Finds the dominant fft component for input (x, y) data."""
    n = x.size
    freqs = np.fft.rfftfreq(n, x[1]-x[0])
    fft = np.fft.rfft(y-np.mean(y))  # TODO: use `fft` rather than `rfft` for complex data.
    freq_guess = freqs[np.argmax(abs(fft))]
    if freq_guess == 0:
        print('Something went wrong, freq guess was zero.')
        freq_guess = 1
    return freq_guess

def find_freq_guess_complex(x, y):
    """Finds the dominant fft component, handling complex data"""
    n = x.size
    freqs = np.fft.fftfreq(n, x[1]-x[0])
    fft = np.fft.fft(y-np.mean(y))
    freq_guess = freqs[np.argmax(abs(fft))]
    return freq_guess

def roundoff(x,roundto):
    """Round x to given precision.
    e.g. roundoff(3.141592653, 1e-3) = 3.142
    """
    return np.round(x/roundto)*roundto

def split_freq(freq, side_range):
    """Split a frequency into side band and LO part.
    
    Args:
        freq: num, frequency to split.
        side_range: tuple, (low_limit, up_limit).

    Returns:
        LO freq, side band freq (in given range).
    """
    low, up = side_range
    side_freq = (freq % (up - low)) + low
    lo_freq = freq - side_freq
    return lo_freq, side_freq

def ratio_to_db(ratio):
    """Voltage ratio -> power db."""
    return 20 * np.log10(ratio)

def db_to_ratio(db):
    """Power db -> voltage ratio."""
    return 10 ** (db / 20)

def amplify(v, gain):
    """Returns voltages amplified by given value."""
    ratio = db_to_ratio(db=gain)
    return v * ratio

def attenuate(v, atten):
    """Returns voltage attenuated by given value."""
    ratio = db_to_ratio(db=-atten)  # +atten = -amp.
    return v * ratio

def multiples(period, shift, vmin, vmax):
    """Returns multiples of period with shift within [vmin, vmax]."""
    nmin = (vmin - shift) // period + 1
    nmax = (vmax - shift) // period
    vs = np.arange(nmin, nmax+1) * period + shift
    return vs

def inverse(func, y, x0=0, **kwargs):
    """Returns f^-1(y).
    
    Args:
        func: callable(x, **kwargs) -> y. 
            Only function of single variable is supported, i.e. no high dimensional system.
            The function would better be monotonic, because for each y, only one x value will be returned.
        y: float or np.array.
            if array, solve f^-1(y) elment by element.
        x0: the initial guess for the inverse function to search x.
    
    Retuns:
        x for which func(x, **kwargs) is close to y.

    Examples:
        def f(x, r):
            return x + r*np.sin(x)
        x = np.linspace(0, 2*np.pi)
        y = f(x, r=1)
        plt.plot(x, y)
        plt.plot(inverse(f, y, r=1), y)  # shold have the same shape as above.
    """
    if isinstance(y, np.ndarray) and isinstance(x0, np.ndarray):
        # Solve the values one by one instead of a high-dimensional system (it is decoupled).
        x = [fsolve(lambda xi: yi - func(xi, **kwargs), x0=x0i)[0] 
            for yi, x0i in (y.ravel(), x0.ravel())]
        x = np.array(x).reshape(y.shape)
    elif isinstance(y, np.ndarray):
        x = [fsolve(lambda xi: yi - func(xi, **kwargs), x0=x0)[0] 
            for yi in y.ravel()]
        x = np.array(x).reshape(y.shape)
    else:
        x = fsolve(lambda xi: y - func(xi, **kwargs), x0=x0)[0]
    return x
