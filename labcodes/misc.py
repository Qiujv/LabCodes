import math
from functools import wraps

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve


def auto_rotate(data, return_rad=False):
    """Returns data with shifted phase s.t. variation of imaginary part minimized.
    
    Algorithm by Kaho.
    """
    rad = -0.5 * np.angle(np.mean(data**2) - np.mean(data)**2)  # Minimize imag(var(data)), by Kaho
    
    if return_rad is True:
        return data * np.exp(1j*rad), rad  # counter-clockwise
    else:
        return data * np.exp(1j*rad)

def remove_e_delay(phase, freq, i_start=0, i_end=-1):
    """Returns phase without linear freq dependence.

    Args:
        phase: real numbers, with linear dependence to freq.
        freq: real numbers with same shape as phase.
        i_start, i_end: region where linear fit applied.
    """
    phase = np.unwrap(phase)
    e_delay = (phase[i_end] - phase[i_start]) / (freq[i_end] - freq[i_start])
    phase -= e_delay * (freq - freq[i_start])
    phase -= phase[i_start]
    return phase

def find_freq_guess(x, y):
    """Finds the dominant fft component for input (x, y) data."""
    if np.iscomplexobj(y):
        fft =  np.fft.fft(y-np.mean(y))
        freqs = np.fft.fftfreq(x.size, x[1]-x[0])
    else:  # usef rfft for real data.
        fft =  np.fft.rfft(y-np.mean(y))
        freqs = np.fft.rfftfreq(x.size, x[1]-x[0])

    freq_guess = freqs[np.argmax(abs(fft))]
    if freq_guess == 0:
        print('Something went wrong, freq guess was zero.')
        freq_guess = 1

    return freq_guess

def round(x, roundto):
    """Round x to given precision.
    e.g. round(3.141592653, 1e-3) = 3.142
    """
    return np.round(x/roundto)*roundto

def start_stop(start, stop, step=None, n=None) -> np.array:
    """Returns evenly space array.
    
    >>> start_stop(1, 2, 0.2)
    array([1. , 1.2, 1.4, 1.6, 1.8, 2. ])

    >>> start_stop(2, 1, n=6)
    array([2. , 1.8, 1.6, 1.4, 1.2, 1. ])

    >>> start_stop(1, 1.9999, 0.2)  # NOTE the unexpected behavior.
    array([1. , 1.2, 1.4, 1.6, 1.8, 2. ])

    >>> start_stop(1, 5, 1)  # Return int type if possible.
    array([1, 2, 3, 4, 5])

    >>> start_stop(1, 5, n=5)
    array([1, 2, 3, 4, 5])
    """
    if n is None: 
        if (
            isinstance(start, int) 
            and isinstance(stop, int) 
            and isinstance(step, int)
        ):
            dtype = int
        else:
            dtype = None
        arr = np.arange(start, stop+step*0.01, step, dtype=dtype)
    else: 
        if (
            isinstance(start, int) 
            and isinstance(stop, int) 
            and ((stop - start) % (n-1) == 0)
        ):
            dtype = int
        else:
            dtype = None
        arr = np.linspace(start, stop, n, dtype=dtype)
    return arr

def center_span(center, span, step=None, n=None) -> np.array:
    """Returns evenly space array.

    >>> center_span(1, 2, 0.4)
    array([0.2, 0.6, 1. , 1.4, 1.8])

    >>> center_span(1, 2, n=5)
    array([0. , 0.5, 1. , 1.5, 2. ])

    >>> center_span(0, 4, 1)  # Return int type if possible.
    array([-2, -1,  0,  1,  2])

    >>> center_span(0, 4, n=5)
    array([-2, -1,  0,  1,  2])
    """
    if n is None: 
        n2 = (span/2) // step
        arr = np.arange(-n2, n2 + 1, dtype=int)
        arr = arr * step + center
    else:
        arr_f = np.linspace(center - span/2, center + span/2, n)
        arr_d = np.linspace(center - span/2, center + span/2, n, dtype=int)
        arr = arr_d if np.allclose(arr_f, arr_d) else arr_f
    return arr


def segments(*segs) -> np.array:
    """Concate multiple segments. Remove repeated endpoints.
    
    >>> segments(
        start_stop(0,1,0.2),
        start_stop(1,10,2),
    )
    array([0. , 0.2, 0.4, 0.6, 0.8, 1. , 3. , 5. , 7. , 9. ])
    """
    segs = list(segs)
    for i in range(len(segs) - 1):
        if np.isclose(segs[i][-1], segs[i+1][0]):
            segs[i+1] = segs[i+1][1:]
    return np.hstack(segs)


def multiples(period, shift, vmin, vmax):
    """Returns multiples of period with shift within [vmin, vmax]."""
    nmin = (vmin - shift) // period + 1
    nmax = (vmax - shift) // period
    vs = np.arange(nmin, nmax+1) * period + shift
    return vs

def simple_interp(x, xp, yp, **kwargs):
    """Wrapper for np.interp but check monoliraty of xp."""
    if np.all(np.diff(xp) > 0):
        return np.interp(x, xp, yp, **kwargs)
    elif np.all(np.diff(xp) < 0):
        return np.interp(x, xp[::-1], yp[::-1], **kwargs)
    else:
        raise ValueError("xp must be monotonic")

def inverse(func, y, x0=None, xlim=None, fast=False, show=False):
    """Returns f^-1(y).
    
    Args:
        func: callable(x) -> y. 
            Only function of single variable is supported, i.e. no high dimensional system.
            The function should be monotonic within xlim.
        y: float or np.array.
            if array, solve f^-1(y) elment by element.
        x0: the initial guess for the inverse function to search x.
        xlim: region where the solution cannot lay beyond. also used for guess x0 from None.
    
    Retuns:
        x for which func(x) is close to y.
    """
    if xlim:
        xlower, xupper = xlim
        if func(xupper) < func(xlower):
            xlower, xupper = xupper, xlower
        func = bound(*np.sort(xlim))(func)
    
    if x0 is None:
        if xlim:
            # Find rough solution with interplotation, for fast fsolve.
            xspace = np.linspace(xlower, xupper, 1000)
            x0 = np.interp(y, func(xspace), xspace)
        else:
            x0 = 0.

    if not isinstance(y, np.ndarray):
        x = fsolve(lambda xi: y - func(xi), x0=x0)[0]
    else:
        x0 = x0 * np.ones(y.shape)
        if fast is True:
            x = x0
            if xlim is None: print('WARNING: fast inverse without xlim simply returns x0.')
        else:
            # NOTE: Could be time-consuming when y.size is large and x0 is bad.
            x = [fsolve(lambda xi: yi - func(xi), x0=x0i)[0] 
                 for yi, x0i in zip(y.ravel(), x0.ravel())]
        x = np.array(x).reshape(y.shape)

    if show is True:
        _, ax = plt.subplots()
        ax.plot(x, y, label='y')  # Assumes right solution found.
        ax.plot(x0, func(x0), 'o', label='init', fillstyle='none')
        ax.plot(x, func(x), 'x', label='find')
        ax.legend()
        plt.show(block=True)

    return x

def _bound(f, x, xleft, xright, scale):
    # f(x) = y, x is scalar.  # BUG: inverse works abnormally when scale != 1.
    if xleft > xright:
        xleft, xright = xright, xleft  # Exchange to make sure xleft <= xright.
    x1, x2 = xleft, xright
    y1, y2 = f(x1), f(x2)
    if x <= x1:
        return (x-x1)*(y2-y1)/(x2-x1)*scale + y1
    elif x >= x2:
        return (x-x2)*(y2-y1)/(x2-x1)*scale + y2
    else:
        return f(x)

def bound(xleft, xright, scale=1., verbose=False):  # A decorator factory.
    """Decorate f(x, *arg, **kwargs) such that f(x) with x beyond [xleft, xright] 
    is obtained by linear extrapolation. Intended for `misc.inverse`.`"""
    def decorator(f):
        @wraps(f)
        def wrapped_f(x, *args, **kwargs):
            def fx(x): return f(x, *args, **kwargs)
            if np.size(x) == 1:
                if verbose:
                    if (x < xleft) or (x > xright):
                        print('WARNING: value {} out of bound [{},{}].'.format(x, xleft, xright))
                return _bound(fx, x, xleft, xright, scale)
            else:
                x = np.array(x)
                if verbose:
                    mask = (x < xleft) | (x > xright)
                    if np.any(mask):
                        print('WARNING: value {} out of bound [{},{}].'.format(x[mask], xleft, xright))
                ys = [_bound(fx, xi, xleft, xright, scale) for xi in x]
                return np.array(ys)
        return wrapped_f
    return decorator

def num2bstr(num, n_bits, base=2):
    if num >= base**n_bits:
        msg = 'num {} requires more than {} bits with base {} to store.'
        raise ValueError(msg.format(num, n_bits, base))
    if base > 10:
        print('WARNING: base > 10 is not implemented yet!')

    l = []
    while True:
        l.append(num % base)  # TODO: consider use np.base_repr
        last_num = num
        num = num // base
        if last_num // base == 0:
            break
    bit_string = ''.join([str(i) for i in l[::-1]])
    return bit_string.zfill(n_bits)

def bitstrings(n_qbs, base=2):
    """Returns ['00', '01', '10', '11'] for n_qbs=2, and etc."""
    return [num2bstr(i, n_qbs, base=base) for i in range(base**n_qbs)]


ENG_PREFIXES = {
    -24: "y",
    -21: "z",
    -18: "a",
    -15: "f",
    -12: "p",
    -9: "n",
    -6: "\N{MICRO SIGN}",
    -3: "m",
    0: "",
    3: "k",
    6: "M",
    9: "G",
    12: "T",
    15: "P",
    18: "E",
    21: "Z",
    24: "Y"
}

def estr(num, places=None, sep=' '):
    """Format a number in engineering notation, appending a letter
    representing the power of 1000 of the original number.

    Adapted from `matplotlib.ticker.EngFormatter`

    Examples:
        >>> eng_string(0, places=0)
        '0'

        >>> eng_string(1000000, places=1)
        '1.0 M'

        >>> eng_string("-1e-6", places=2)
        '-1.00 \N{MICRO SIGN}'

    Args: 
        places : int, default: None
            Precision with which to display the number, specified in
            digits after the decimal point (there will be between one
            and three digits before the decimal point). If it is None,
            the formatting falls back to the floating point format '%g',
            which displays up to 6 *significant* digits, i.e. the equivalent
            value for *places* varies between 0 and 5 (inclusive).

        sep : str, default: " "
            Separator used between the value and the prefix/unit. For
            example, one get '3.14 mV' if ``sep`` is " " (default) and
            '3.14mV' if ``sep`` is "". Besides the default behavior, some
            other useful options may be:

            * ``sep=""`` to append directly the prefix/unit to the value;
            * ``sep="\N{THIN SPACE}"`` (``U+2009``);
            * ``sep="\N{NARROW NO-BREAK SPACE}"`` (``U+202F``);
            * ``sep="\N{NO-BREAK SPACE}"`` (``U+00A0``).

    Returns: 
        String of the formatted num.

    Notes:
        To use this in axis ticks,
        ```python
        from matplotlib.ticker import EngFormatter
        ax.xaxis.set_major_formatter(EngFormatter(unit='Hz'))
        ```
    """
    sign = 1
    fmt = "g" if places is None else ".{:d}f".format(places)

    if num < 0:
        sign = -1
        num = -num

    if num != 0:
        pow10 = int(math.floor(math.log10(num) / 3) * 3)
    else:
        pow10 = 0
        # Force num to zero, to avoid inconsistencies like
        # format_eng(-0) = "0" and format_eng(0.0) = "0"
        # but format_eng(-0.0) = "-0.0"
        num = 0.0

    pow10 = np.clip(pow10, min(ENG_PREFIXES), max(ENG_PREFIXES))

    mant = sign * num / (10.0 ** pow10)
    # Taking care of the cases like 999.9..., which may be rounded to 1000
    # instead of 1 k.  Beware of the corner case of values that are beyond
    # the range of SI prefixes (i.e. > 'Y').
    if (abs(float(format(mant, fmt))) >= 1000
            and pow10 < max(ENG_PREFIXES)):
        mant /= 1000
        pow10 += 3

    prefix = ENG_PREFIXES[int(pow10)]
    formatted = "{mant:{fmt}}{sep}{prefix}".format(
        mant=mant, sep=sep, prefix=prefix, fmt=fmt)

    return formatted

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    def f(x, r):
        return x + r*np.sin(x)
    x = np.linspace(0, 2*np.pi)
    y = f(x, r=1)
    plt.plot(x, y, label='original one')
    plt.plot(inverse(f, y, r=1), y, label='inversed one')  # shold have the same shape as above.
    plt.legend()
    plt.show()
