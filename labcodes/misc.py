"""Module contains miscellaneous useful functions.
"""

import numpy as np
from pathlib import Path


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

def search_file(dir, keyword):
    """Search specified file in given directory.
    
    Args:
        dir, directory to search, including subdirectories of it.
        keyword, str, search keywor.

    Returns:
        list(pathlib.Path), path of files with the keyword in name, sorted in descending order.
    """
    return sorted(list(Path(dir).resolve().glob(f'**/*{keyword}*')), reverse=True)

def get_chn(instr, channel: str):
    """Get channel name from specific Labber.Instrument.
    
    Args:
        instr, Labber.Instrument.
        channle, str, name of instrument channel.
        
    Returns:
        str, the full channel name.
    """
    return instr.com_config.name + ' - ' + channel

def search_channel(instr, keyword:str):
    """Search channel with given keywords among the instrument.
    
    Args:
        instr, Labber.Instrument.
        keyword, str.

    Returns:
        dict, channels containing keyword and their values.
    """
    return {key: value for key, value in instr.values.items() if keyword in key}

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