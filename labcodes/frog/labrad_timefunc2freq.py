"""Generate envelope with arbitrary shape. `freq_func` is calculated by FFT and linear interplotation.


Script provided by Ziyu Tao, at 2022.4.17
"""
# Change Log:
# 2022.4.17, Jiawei Qiu, change name convention to Google's style.

# %%
from matplotlib import pyplot as plt
import numpy as np


def test_pulse(t0=0, width=10, plateau=20, amp=1):

    def mask(t, tmin, tmax):
        """Return 1 if tmin < t < tmax, 0 otherwise."""
        return (np.sign(tmax-t) + np.sign(t-tmin)) / 2.

    if width == 0:
        t1 = t0 + plateau
        if plateau == 0:
            def time_func(t):
                return np.zeros_like(t)
        else:
            def time_func(t):
                return 1*(t <= t1)*(t >= t0)
    else:
        hw = width / 2.  # Half width
        t1 = t0 + width + plateau
        def time_func(t):
            t = t-t0
            pt_rise = np.sin(t/hw*np.pi/2.) * mask(t, 0., hw)
            pt_fall = np.sin((t-hw-plateau)/hw*np.pi/2. + np.pi/2.) * mask(t, hw+plateau, 2.*hw+plateau)
            pt_plat = 1. * mask(t, hw, hw+plateau)
            return amp*(pt_rise + pt_plat + pt_fall)

    def freq_func(f):
        # Pad time series for dense frequency points.
        pad_len = 200000
        vt = np.arange(t0-pad_len, t1+pad_len)
        # Round size padded time series to power of 2, for sake of faster FFT.
        n_pts = 2**int(np.ceil(np.log2(vt.size)))
        margin = (n_pts-np.ceil(t1-t0)) / 2.
        pre_pad_len = int(np.ceil(margin))
        post_pad_len = int(np.floor(margin))
        vt = np.arange(t0-pre_pad_len, t1+post_pad_len)
        assert np.ceil(np.log2(vt.size)) == np.log2(vt.size)

        # FFT and linear interpolation.
        vy = time_func(vt)
        freq = np.fft.fftfreq(n_pts)
        vf = np.fft.fft(vy)*np.exp(-2j*np.pi*(t0-pre_pad_len)*freq)
        idx = np.argsort(freq)
        freq = freq[idx]
        vf = vf[idx]
        return np.interp(f,freq,vf.real) + 1j*np.interp(f,freq,vf.imag)
        
    # return Envelope(time_func, freq_func, start=t0, end=t1)
    return time_func, freq_func


if __name__ == '__main__':
    # time_func, freq_func = test_pulse(t0=0,length = 140)
    time_func, freq_func = test_pulse(t0=700.8, width=0, plateau=314.2, amp=0.2)
    ts = np.arange(-10, 1000, 1)
    fs = np.arange(-0.5, 0.5, 0.001)
    ys = time_func(ts)
    ys_f = freq_func(fs)
    plt.plot(ts, ys)
    plt.figure()
    plt.plot(fs, np.real(ys_f))
    plt.plot(fs, np.imag(ys_f))

# %%
