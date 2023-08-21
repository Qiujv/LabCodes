import functools
import logging
import math
from collections.abc import Hashable
from typing import Optional, Union

import numpy as np
import scipy.interpolate

logger = logging.getLogger(__name__)


def fidelity(a: np.ndarray, b: np.ndarray):
    """Returns fidelity between two matrices.

    >>> fidelity([[1,0],[0,0]], [[.5,.5],[.5,.5]])
    0.5
    """
    logger.critical("This function is wrong, use tomo.fidelity instead.")
    a, b = np.asarray(a), np.asarray(b)
    return np.real(np.trace(a @ b))


def auto_rotate(data: np.ndarray[complex], return_rad: bool = False):
    """Returns data with shifted phase s.t. variation of imaginary part minimized.
    
    >>> rot_data = auto_rotate([0, 1j])
    >>> rot_data.real
    array([0., 1.])
    >>> np.allclose(rot_data.imag, 0)
    True
    """
    data = np.asarray(data)

    # Minimize imag(var(data)), by Kaho
    rad = -0.5 * np.angle(np.mean(data**2) - np.mean(data) ** 2)

    if return_rad is True:
        return data * np.exp(1j * rad), rad  # counter-clockwise
    else:
        return data * np.exp(1j * rad)


def remove_e_delay(phase, freq, i_start=0, i_end=-1):
    """Returns phase without linear freq dependence.

    Args:
        phase: array of angle in rads, with linear dependence to freq.
        freq: array of same shape as phase.
        i_start, i_end: region where linear fit applied.

    >>> remove_e_delay([0, np.pi], [0, 1])
    array([0., 0.])
    """
    phase = np.asarray(phase)
    freq = np.asarray(freq)

    phase = np.unwrap(phase)
    e_delay = (phase[i_end] - phase[i_start]) / (freq[i_end] - freq[i_start])
    e_phase = e_delay * (freq - freq[i_start]) + phase[i_start]
    return phase - e_phase


def guess_freq(x: np.ndarray[float], y: np.ndarray[float]) -> float:
    """Finds the dominant fft component for input (x, y) data.

    Assumes x to by evenly spaced.

    >>> x = np.linspace(0, 1, 101)
    >>> freq = 2
    >>> guess_freq(x, np.sin(2*np.pi*freq*x))
    1.9801980198019802

    >>> x = np.linspace(0, 1, 51)  # A bit too few points.
    >>> guess_freq(x, np.sin(2*np.pi*freq*x))
    1.9607843137254901
    """
    if np.iscomplexobj(y):
        fft = np.fft.fft(y - np.mean(y))
        freqs = np.fft.fftfreq(x.size, x[1] - x[0])
    else:  # usef rfft for real data.
        fft = np.fft.rfft(y - np.mean(y))
        freqs = np.fft.rfftfreq(x.size, x[1] - x[0])

    freq_guess = freqs[np.argmax(abs(fft))]
    if freq_guess == 0:
        print("Something went wrong, freq guess was zero.")
        freq_guess = 1

    return freq_guess


find_freq_guess = guess_freq  # Old name, for backward compatibility.


def guess_phase(
    x: np.ndarray,
    y: np.ndarray,
    freq: float = None,
    phi_space: np.ndarray = None,
) -> float:
    """Return the possible phase for single-frequency, noisy data.

    Args:
        phi_space: the returned phi will be one in this.
            if None, use `np.linspace(-np.pi, np.pi, 51)`

    Examples:
    >>> x = np.linspace(0, 1, 101)
    >>> freq, phase = 2, 1
    >>> np.random.seed(0)
    >>> y = np.sin(2*np.pi*freq*x + phase) + np.random.uniform(-1, 1, x.size)*0.05
    >>> guess_phase(x, y)
    1.0681415022205298

    >>> x = np.linspace(0, 1, 51)  # A bit too few points.
    >>> freq, phase = 2, 1
    >>> y = np.sin(2*np.pi*freq*x + phase) + np.random.uniform(-1, 1, x.size)*0.05
    >>> guess_phase(x, y)
    1.1309733552923262

    >>> guess_phase(x, y, freq)  # Given accurate freq. makes it robust.
    1.0053096491487343
    """
    if freq is None:
        freq = find_freq_guess(x, y)  # Could be inaccurate.
    if phi_space is None:
        phi_space = np.linspace(-np.pi, np.pi, 101)
    integral = [np.sum(y * np.sin(2 * np.pi * freq * x + phase)) for phase in phi_space]
    imax = np.argmax(integral)
    return phi_space[imax]


def round(x: float, roundto: float) -> float:
    """Round x to given precision.

    >>> round(3.141592653, 1e-3)
    3.142
    """
    return np.round(x / roundto) * roundto


def start_stop(start, stop, step=None, n=None) -> np.ndarray:
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
        if (start > stop) and (step > 0):
            logger.warning("start > stop, but step > 0, use step = -step instead.")
            step = -step
        if isinstance(start, int) and isinstance(stop, int) and isinstance(step, int):
            dtype = int
        else:
            dtype = None
        arr = np.arange(start, stop + step * 0.01, step, dtype=dtype)
    else:
        if (
            isinstance(start, int)
            and isinstance(stop, int)
            and ((stop - start) % (n - 1) == 0)
        ):
            dtype = int
        else:
            dtype = None
        arr = np.linspace(start, stop, n, dtype=dtype)
    return arr


def center_span(center, span, step=None, n=None) -> np.ndarray:
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
        n2 = (span / 2) // step
        arr = np.arange(-n2, n2 + 1, dtype=int)
        arr = arr * step + center
    else:
        arr_f = np.linspace(center - span / 2, center + span / 2, n)
        arr_d = np.linspace(center - span / 2, center + span / 2, n, dtype=int)
        arr = arr_d if np.allclose(arr_f, arr_d) else arr_f
    return arr


def segments(*segs) -> np.ndarray:
    """Concate multiple segments. Remove repeated endpoints.

    >>> segments(
    ...     start_stop(0, 1, 0.2),
    ...     start_stop(1, 10, 2),
    ... )
    array([0. , 0.2, 0.4, 0.6, 0.8, 1. , 3. , 5. , 7. , 9. ])
    """
    segs = list(segs)
    for i in range(len(segs) - 1):
        if np.isclose(segs[i][-1], segs[i + 1][0]):
            segs[i + 1] = segs[i + 1][1:]
    return np.hstack(segs)


def zigzag_arange(n):
    """Returns indices that picks the first and last first, and converges to center in the end.

    >>> zigzag_arange(7)
    array([0, 6, 1, 5, 2, 4, 3])

    >>> center_span(0, 6, n=7)[zigzag_arange(7)]
    array([-3,  3, -2,  2, -1,  1,  0])
    """
    idx = np.c_[np.arange(n // 2), n - 1 - np.arange(n // 2)].ravel()
    if n % 2 != 0:
        idx = np.r_[idx, n // 2]
    return idx


def multiples(period, shift, vmin, vmax) -> np.ndarray:
    """Returns multiples of period with shift within [vmin, vmax].

    >>> multiples(1, 0.1, 0, 5)
    array([0.1, 1.1, 2.1, 3.1, 4.1])
    """
    nmin = (vmin - shift) // period + 1
    nmax = (vmax - shift) // period
    vs = np.arange(nmin, nmax + 1) * period + shift
    return vs


def simple_interp(x, xp, yp, **kwargs):
    """Wrapper for np.interp but check monoliraty of xp.

    >> simple_interp(0.3, [1,0], [1,0])
    0.3
    """
    dx = np.diff(xp)
    if np.all(dx > 0):
        return np.interp(x, xp, yp, **kwargs)
    elif np.all(dx < 0):
        return np.interp(x, xp[::-1], yp[::-1], **kwargs)
    else:
        raise ValueError("xp must be monotonic")


def inverse_interp(
    func: callable,
    y: Union[np.ndarray, float],
    xp: np.ndarray,
    tol=1e-6,
) -> Union[np.ndarray, float]:
    """Calculate values of x that gives y=f(x) with interpolation.

    Requires f(xp) being monolithic.

    >>> inverse_interp(lambda x: x**2, 4, np.linspace(0, 5, 10000))
    1.9999999849966246

    >>> inverse_interp(lambda x: x**2, [1, 4], np.linspace(0, 5, 10000))
    array([0.99999998, 1.99999998])
    """
    is_scalar = np.isscalar(y)
    xp, y = np.asarray(xp), np.asarray(y)
    yp = func(xp)
    _dy = np.diff(yp)
    if np.all(_dy < 0):
        yp, xp = yp[::-1], xp[::-1]
    if np.any(_dy <= 0):
        raise ValueError("f(xp) must be monolithic.")
    finv = scipy.interpolate.UnivariateSpline(yp, xp, k=1, s=0)
    x = finv(y)

    # Check extrapolation.
    mask_low = y < yp[0]
    mask_high = y > yp[-1]
    msgs = ["Extrapolation detected. "]
    if np.any(mask_low):
        msgs.append(
            f"{np.sum(mask_low)} points are below the lower bound f({xp[0]})={yp[0]}"
            f" by:\n{(y[mask_low] - yp[0])[:10]}"
        )
    if np.any(mask_high):
        msgs.append(
            f"{np.sum(mask_high)} points are above the upper bound f({xp[-1]})={yp[-1]}"
            f" by:\n{(y[mask_high] - yp[-1])[:10]}"
        )
    if len(msgs) > 1:
        logging.warning("\n".join(msgs))

    # Check tolerance.
    mask_extrap = mask_low | mask_high
    mask_tol = np.abs(y[~mask_extrap] - func(x[~mask_extrap])) < tol
    if not np.all(mask_tol):
        logging.warning(
            f"Failed to find inverse for {np.sum(~mask_tol)} points"
            f" (except extrapolate) within tolerance {tol}."
        )
    
    if is_scalar:
        return x.item()
    else:
        return x


def num2bstr(num: int, n_bits: int, base: int = 2) -> str:
    """Converts a number to a bit string with leading zeros.

    >>> num2bstr(3, 4)
    '0011'
    """
    new = np.base_repr(num, base).zfill(n_bits)
    old = _old_num2bstr(num, n_bits, base)
    if new == old:
        return new
    else:
        logger.warning(
            "Inconsistence found, "
            f"_old_num2bstr(num={num}, n_bits={n_bits}, base={base}) = {old}, "
            f"np.base_repr({num}, {base}).zfill({n_bits}) = {new}.",
            stack_info=True,
            stacklevel=2,
        )
        return old

def _old_num2bstr(num: int, n_bits: int, base: int = 2) -> str:
    if num >= base**n_bits:
        msg = "num {} requires more than {} bits with base {} to store."
        raise ValueError(msg.format(num, n_bits, base))
    if base > 10:
        logger.warning("base > 10 is not implemented yet!")

    l = []
    while True:
        l.append(num % base)
        last_num = num
        num = num // base
        if last_num // base == 0:
            break
    bit_string = "".join([str(i) for i in l[::-1]])
    return bit_string.zfill(n_bits)


def bitstrings(n_qbs, base=2):
    """Returns bit strings of n_qbs qubits with the base.

    >>> bitstrings(2)
    ['00', '01', '10', '11']

    >>> bitstrings(2, base=3)
    ['00', '01', '02', '10', '11', '12', '20', '21', '22']

    >>> bitstrings(3)
    ['000', '001', '010', '011', '100', '101', '110', '111']
    """
    return [num2bstr(i, n_qbs, base=base) for i in range(base**n_qbs)]


def estr(num: float, places: int = None, sep: str = "") -> str:
    """Format a number in engineering notation, appending a letter
    representing the power of 1000 of the original number.

    Adapted from `matplotlib.ticker.EngFormatter`

    Examples:
        >>> estr(0, places=0)
        '0'

        >>> estr(1000000, places=1)
        '1.0M'

        >>> estr(-1e-6, places=2)
        '-1.00µ'

    Args:
        places: Precision with which to display the number If None, displays up
            to 6 *significant* digits.

        sep : Separator used between the value and the prefix/unit.
            could be unicode like `"\\u2009"` (thin space), `\\u202f` (narrow
            no-break space) or `"\\u00a0"` (no-break space).

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

    mant = sign * num / (10.0**pow10)
    # Taking care of the cases like 999.9..., which may be rounded to 1000
    # instead of 1 k.  Beware of the corner case of values that are beyond
    # the range of SI prefixes (i.e. > 'Y').
    if abs(float(format(mant, fmt))) >= 1000 and pow10 < max(ENG_PREFIXES):
        mant /= 1000
        pow10 += 3

    prefix = ENG_PREFIXES[int(pow10)]
    return f"{mant:{fmt}}{sep}{prefix}"


ENG_PREFIXES = {
    -24: "y",
    -21: "z",
    -18: "a",
    -15: "f",
    -12: "p",
    -9: "n",
    -6: "µ",
    -3: "m",
    0: "",
    3: "k",
    6: "M",
    9: "G",
    12: "T",
    15: "P",
    18: "E",
    21: "Z",
    24: "Y",
}


def remove_background(df, x_name, y_name, z_name):
    """Remove the background of Z which varies along X and constant along Y.
    yyyyyy
    Assumes rectangle grid sampling.

    Example:
        remove_background(lf.df, 'ro_freq_GHz', 'z_pulse_offset', ['iq_amp', 'abs(s21)_dB', 'iq_phase_rad'])
    """
    background = df.groupby(x_name)[z_name].mean()
    def trans(df):
        df[z_name] = df[z_name].values - background.values
        return df
    return df.groupby(y_name).apply(trans).reset_index(drop=True)


def cache_with_bypass(
    maxsize: Optional[int] = 128,
    typed: bool = False,
    warn_if_bypass: bool = False,
):
    """Cache a function with functools.lru_cache.

    If the arguments are not hashable, bypass the cache and run the function instead of
    raising TypeError like functools.lru_cache does.

    >>> @cache_with_bypass()
    ... def func(a):
    ...     print("calculating", a)
    ...     return a ** 2
    >>> for i in range(3):
    ...     func(i)
    calculating 0
    0
    calculating 1
    1
    calculating 2
    4
    >>> for i in range(3):
    ...     func(i)
    0
    1
    4
    >>> func.cached.cache_info()
    CacheInfo(hits=3, misses=3, maxsize=128, currsize=3)
    """
    log_level = logging.WARNING if warn_if_bypass else logging.INFO
    def decorator(func):
        @functools.wraps(func)
        def wrapped_f(*args, **kwargs):
            if any(not isinstance(arg, Hashable) for arg in args) or any(
                not isinstance(val, Hashable) for val in kwargs.values()
            ):
                logger.log(
                    log_level,
                    "Arguments for %s is not hashable. Cache is disabled.",
                    func.__name__,
                )
                return func(*args, **kwargs)
            else:
                return wrapped_f.cached(*args, **kwargs)

        wrapped_f.cached = functools.lru_cache(maxsize=maxsize, typed=typed)(func)
        return wrapped_f

    return decorator


if __name__ == "__main__":
    import doctest

    doctest.testmod()
