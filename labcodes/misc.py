"""Module contains miscellaneous useful functions.
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from pathlib import Path


def phase_rotate(data, angle=None, show=False):
    """Rotate data clockwise with angle in rad.
    
    Args:
        data: np.array.
        angle: float, phase angle to rotate.
            if None, estimated value by get_rotate_angle will be admitted.
        show: boolean,
            Whether or not to plot the rotated data in complex plane.
            Default is False.
    
    Returns:
        data * np.exp(1j*angle)
    """
    data = np.array(data)
    if angle is None: angle = get_rotate_angle(data)

    rot_data = data * np.exp(1j*angle) # counter-clockwise
    
    if show:
        show_data = rot_data.ravel()
        n_pt_max = 3000
        fig, ax = plt.subplots(tight_layout=True)
        if show_data.size <= n_pt_max:
            ax.plot(show_data.real, show_data.imag, 
                '.', alpha=0.7)
        else:
            ax.hist2d(show_data.real, show_data.imag, 
                bins=100, norm=mcolors.PowerNorm(0.5))

        ax.set(
            title=f'Data rotated by {angle*180/np.pi:+.2f} (CCW)',
            xlabel='real',
            ylabel='imag',
        )
        ax.ticklabel_format(scilimits=(-2,4))
        
        plt.show()
        plt.close(fig)
    return rot_data

def get_rotate_angle(data):
    """Returns phase angle (in rad) that minize imaginary part of data. 
    Algorithm by Kaho."""
    # xdata = np.real(data.ravel())
    # ydata = np.imag(data.ravel())
    # slope1 = np.polyfit(xdata, ydata, 1)[0]
    # slope2 = np.polyfit(ydata, xdata, 1)[0]
    # # find the one further from x axis in fitting.
    # if abs(slope1) < abs(slope2):
    #     angle = np.arctan(slope1)
    # else:
    #     angle = np.pi/2 - np.arctan(slope2)
    # # get the angle to rotate
    # rot_angle = -angle % np.pi
    angle = -0.5 * np.angle(np.mean(data**2) - np.mean(data)**2)
    return angle # in rad

# def phase_rotate_x(data, x, rot_rate):  # TODO: Remove this.
#     """Returns data with additional phase propotional to x value."""
#     return data*np.exp(1j*x*rot_rate)

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

def mid_every_pair(v, expand=False):
    """v: [1,2,3,4] -> mv: [1.5, 2.5, 3.5] or [0.5, 1.5, 2.5, 3.5, 4.5]."""
    v = np.array(v)
    mv = (v[1:] + v[:-1]) / 2
    if expand is True:
        head = 2 * v[0] - mv[0]
        mv = np.insert(mv, 0, head)
        tail = 2 * v[-1] - mv[-1]
        mv = np.append(mv, tail)
        return mv
    elif expand is False:
        return mv
    else:
        raise TypeError(f'Except boolean `expand`, while {type(expand)} is given.')

def plot_2d(ax, x, y, z, data, **kw):
    """Plot z in colormap versus x and y.
    
    Args:
        ax: matplotlib.axes.Axe, where to plot the artiest.
        x, y, z: str, key of the quantites to plot with.
        data: pandas.DataFrame.
        **kw: other kwargs passed to ax.pcolormesh()

    Returns:
        matplotlib.collections.QuadMesh.
    """
    table = data[[x, y, z]].pivot(index=y, columns=x)
    vx = table.columns.levels[1].values
    vy = table.index.values
    mx = mid_every_pair(vx, expand=True)
    my = mid_every_pair(vy, expand=True)
    return ax.pcolormesh(mx, my, table.values, **kw)