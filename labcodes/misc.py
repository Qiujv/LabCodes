import numpy as np
from scipy.optimize import fsolve
from functools import wraps


def auto_rotate(data, with_angle=False):
    """Returns data with shifted phase s.t. variation of imaginary part minimized.
    Algorithm by Kaho."""
    angle = -0.5 * np.angle(np.mean(data**2) - np.mean(data)**2)  # Minimize imag(var(data)), by Kaho
    
    if with_angle is True:
        return data * np.exp(1j*angle), angle  # counter-clockwise
    else:
        return data * np.exp(1j*angle)

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
            The function should be monotonic, otherwise consider decorate it with `misc.bound`.
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

def _bound(f, x, xleft, xright, scale):
    # f(x) = y, x is scalar.
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

def bound(xleft, xright, scale=1):  # A decorator factory.
    """Decorate f(x, *arg, **kwargs) such that f(x) with x beyond [xleft, xright] 
    is obtained by linear extrapolation. Intended for `misc.inverse`.`"""
    def decorator(f):
        @wraps(f)
        def wrapped_f(x, *args, **kwargs):
            def fx(x): return f(x, *args, **kwargs)
            if np.size(x) == 1:
                return _bound(fx, x, xleft, xright, scale)
            else:
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