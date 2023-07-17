import logging

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline


logger = logging.getLogger(__name__)
pi = np.pi


def soft_edge(t, alpha=1):
    """erf like function, for sech shape a_out. Sattles beyond [-10, 10].
    
    >>> soft_edge(np.array([-10,-1,0,1,10]))
    array([4.53978687e-05, 2.68941421e-01, 5.00000000e-01, 7.31058579e-01,
       9.99954602e-01])
    """
    return np.exp(t) / (1 + np.exp(t)) * alpha / (1 + (1-alpha)*np.exp(t))

def soft_edges_sample(t0, t1, w, alpha=1, cutoff_w=10, sample_rate=1, 
                      edge='both', pwlin=True, mute_in_mid=True, plot=False):
    """Returns sampling points of soft_edge.
    
    >>> vt, vy = soft_edges_sample(0, 300, 10, plot=True, pwlin=False)
    """
    ex = np.linspace(-w*cutoff_w, w*cutoff_w, int(2*w*cutoff_w*sample_rate))
    ey = soft_edge(ex/w, alpha=alpha)  # Pulse sattles within [t0-10w,t1+10w]

    rex = ex + t0
    rey = ey
    fex = t1 - ex[::-1]
    fey = ey[::-1]

    if edge == 'both':
        if (fex[0] - rex[-1]) < 10:
            rex, rey = [i[rex <  (t0+t1)/2] for i in (rex, rey)]
            fex, fey = [i[fex >= (t0+t1)/2] for i in (fex, fey)]
            px = np.hstack([rex, fex])
            py = np.hstack([rey, fey])
        else:
            if mute_in_mid:
                px = np.hstack([rex, rex[-1]+1, fex[0]-1, fex])
                py = np.hstack([rey, 0, 0, fey])
            else:
                px = np.hstack([rex, fex])
                py = np.hstack([rey, fey])
    elif edge == 'rise':
        px = np.hstack((rex, rex[-1]+1))
        py = np.hstack((rey, 0))
    elif edge == 'fall':
        px = np.hstack((fex[0]-1, fex))
        py = np.hstack((0, fey))
    else:
        raise ValueError(edge)
    
    # shift, pad smaple points to produce parameter for pwlin.
    pxs = (px[:-1] + px[1:]) / 2.  # Shift points from center to left of steps.
    pys = py[1:]
    d = pxs[0] - px[0]
    pxs = np.hstack(([pxs[0]-d], pxs, [pxs[-1]+d]))  # Pad the end points.
    pys = np.hstack(([py[0]], pys, [py[-1]]))

    if plot:
        # Check plot.
        fig, ax = plt.subplots()
        ax.set(
            title='soft_edges_sample',
            xlabel='time (ns)',
            ylim=(-0.05,1.05),
        )
        ax.grid(True)
        ax.step(px, py, marker='x', where='mid', color='k', alpha=0.5, lw=1, label='ideal')
        ax.step(pxs[:-1], pys[:-1], marker='.', where='post', label='DAC output')
        ax.axvline(t0, color='k', ls='--')
        ax.axvline(t1, color='k', ls='--')
        ax.legend()
        plt.show()

    if pwlin:
        t_start = pxs[0]
        vdt = np.diff(pxs)
        vt = pxs[:-1]
        vy = pys[:-1]
        return t_start, vdt, vt, vy
    else:
        return px, py

class GmonSpec:
    """Gmon spectrum model.
    
    >>> gfit = GmonSpec(r=0.8, period=1, shift=-0.3, amp=0.005, slope=0, offset=0)
    ... delta_ext = np.linspace(-pi, pi, 51)
    ... delta = gfit.junc_phase(delta_ext)
    ... check = delta + np.sin(delta) * gfit.r
    ... plt.plot(delta, delta_ext, '.')
    ... plt.plot(delta, check, '.')
    ... np.allclose(delta_ext, check, atol=1e-6)
    True

    >>> kappa = np.linspace(0, gfit._kappa_vs_gpa[0][0], 101)
    ... gpa = gfit.gpa_from_kappa(kappa)
    ... check = gfit.fshift_sqr(gpa)
    ... plt.plot(gpa, kappa, '.')
    ... plt.plot(gpa, check, 'k-')
    ... np.allclose(kappa, check, atol=1e-8)
    True
    """
    def __init__(self, r, period, shift, amp, slope, offset):
        self.r = r
        self.period = period
        self.shift = shift
        self.amp = amp
        self.slope = slope
        self.offset = offset
        self._ext_vs_delta = self._calc_ext_vs_delta()
        # self.modify_goff(0)
        self._fshift0 = 0
        self._kappa_vs_gpa = self._calc_kappa_vs_gpa()
    
    def junc_phase(self, delta_ext, tol=1e-6):
        delta = np.interp(delta_ext, *self._ext_vs_delta, period=4*pi)
        mask = _check_extropolate('delta_ext', delta_ext, -2*pi, 2*pi)
        if tol:
            _check_within_tol('delta_ext', delta_ext, 
                              delta + np.sin(delta) * self.r, tol, mask)
        return delta
    
    def _calc_ext_vs_delta(self, n=10001):
        delta = np.linspace(-2*pi, 2*pi, n)  # More periods to cover all possible values.
        delta_ext = delta + np.sin(delta) * self.r  # RF SQUID bias, from Satzinger's thesis.
        return delta_ext, delta

    def _fshift(self, gpa):
        delta_ext = 2*pi * (gpa - self.shift) / self.period
        delta = self.junc_phase(delta_ext)
        M = 1 / (self.r + 1/np.cos(delta))  # Reaches max when gpa=shift.
        return M * self.amp  # Returns 0 if gpa closes the gmon.
    
    def fshift(self, gpa):
        return self._fshift(gpa) - self._fshift0

    def fshift_sqr(self, gpa):
        fshift = self.fshift(gpa)
        sign = np.sign(fshift)
        return sign * fshift**2  # Propotional to kappa.

    def fshift_xtalk(self, gpa):
        fshift = self.fshift(gpa)
        xtalk = gpa * self.slope + self.offset
        return fshift + xtalk

    def gpa_from_kappa(self, kappa, tol=1e-8):
        # gpa = np.interp(kappa, *self._kappa_vs_gpa)
        # Spline handles extropolating case.
        gpa = UnivariateSpline(*self._kappa_vs_gpa, k=1, s=0)(kappa)
        mask = _check_extropolate('kappa', kappa, self._kappa_vs_gpa[0][0], 
                                  self._kappa_vs_gpa[0][-1])
        if tol:
            _check_within_tol('kappa', kappa, self.fshift_sqr(gpa), tol, mask)

        return gpa
    
    def _calc_kappa_vs_gpa(self, n=10001):
        gpa_dip, gpa_top = self.calc_gpa_dip_top()
        gpa = np.linspace(gpa_dip, gpa_top, n)
        kappa = self.fshift_sqr(gpa)
        return kappa, gpa
    
    def calc_gpa_dip_top(self):
        """Returns gpa where the dip and top of gmon spectrum lays."""
        gpa_top = self.shift % self.period  # Where fshift reaches max.
        gpa_dip = gpa_top - self.period/2  # Where fshift reaches min.
        if gpa_dip > 0:  # Make sure [gpa_dip, gpa_top] include gpa=0.
            gpa_top = gpa_top - self.period
        return gpa_dip, gpa_top

    
    def modify_goff(self, gpa):
        """Makes self.fshift returns 0 if gpa=goff."""
        self._fshift0 = self._fshift(gpa)

    
class QubitSpec:
    """Transmon spectrum model.
    
    >>> qfit = QubitSpec(fmin=3.8, fmax=4.6, xmin=-0.1, xmax=0.7)
    ... freq = np.linspace(3.8, 4.6, 101)
    ... zpa = qfit.zpa_from_freq(freq)
    ... check = qfit.freq(zpa)
    ... plt.plot(zpa, freq, '.')
    ... plt.plot(zpa, check, 'k-')
    ... np.allclose(freq, check, atol=1e-8)
    True
    """
    def __init__(self, fmax, fmin, xmax, xmin):
        self.fmax = fmax
        self.fmin = fmin
        self.xmax = xmax
        self.xmin = xmin
        self._f_vs_zpa = self._calc_f_vs_zpa()

    def freq(self, zpa):
        """Frequency of transmon, following koch_charge_2007 Eq.2.18."""
        # Rescale [xmax, xmin] to [0,0.5], i.e. in Phi_0.
        phi = 0.5 * (zpa - self.xmax) / (self.xmin - self.xmax)
        d = (self.fmin / self.fmax) ** 2
        f = self.fmax * np.sqrt(np.abs(np.cos(pi*phi))
                                * np.sqrt(1 + d**2 * np.tan(pi*phi)**2))
        return f
    
    def _calc_f_vs_zpa(self, n=10001):
        zpa = np.linspace(self.xmin, self.xmax, n)
        f = self.freq(zpa)
        return f, zpa

    def zpa_from_freq(self, f, tol=1e-8):
        # zpa = np.interp(f, *self._f_vs_zpa)
        # Spline handles extropolating case.
        zpa = UnivariateSpline(*self._f_vs_zpa, k=1, s=0)(f)
        mask = _check_extropolate('freq', f, self._f_vs_zpa[0][0],
                                  self._f_vs_zpa[0][-1])
        if tol:
            _check_within_tol('freq', f, self.freq(zpa), tol, mask)
        return zpa
    

def _check_extropolate(name, val, min, max):
    val = np.array(val)  # In case val is a scalar.
    mask = (val < min) | (val > max)
    if np.any(mask):
        logger.warning(f'Extrapolating for {name}={val[mask]}', stacklevel=2)
    return mask

def _check_within_tol(name, val, target, tol, mask=None):
    if mask is not None:
        val = np.array(val)[mask]  # In case val is a scalar.
        target = np.array(target)[mask]
    if np.allclose(val, target, atol=tol) is False:
        logger.info(f'inverse {name} out of tol={tol}', stacklevel=2)


if __name__ == '__main__':
    # # Test speed.
    # from viztracer import VizTracer
    # tracer = VizTracer()
    # tracer.start()
    # gfit = GmonSpec(r=0.8, period=1, shift=-0.3, amp=0.005, slope=0, offset=0)
    # kappa = np.linspace(0, gfit._kappa_vs_gpa[0][0] - 1e-4, 101)
    # gpa = gfit.gpa_from_kappa(kappa)
    # check = gfit.fshift_sqr(gpa)
    # np.allclose(kappa, check, atol=1e-6)
    # tracer.stop()
    # tracer.save(r"C:\Users\qiujv\Downloads\spl.html")

    # Test waveform generation.
    def ping_pong_z(  # For simulation purpose.
        gfit:GmonSpec, qfit:QubitSpec, 
        gpa, width, t0, delay,
        goff=None, 
        zpa=0, fs_scale=1,
        alpha=1, edge='both',
        cutoff_w=10.0, 
    ):
        if goff: gfit.modify_goff(goff)
        vt, vy = soft_edges_sample(t0, t0+delay, width, edge=edge, alpha=alpha,
                                   pwlin=False, sample_rate=1, cutoff_w=cutoff_w)
        # kmin = gfit.fshift_sqr(0)
        kmin = 0
        kmax = gfit.fshift_sqr(gpa)
        vks = vy*(kmax-kmin) + kmin
        vgz = gfit.gpa_from_kappa(vks)
        vdf = gfit.fshift_xtalk(vgz) - gfit.fshift_xtalk(0)
        vqz = qfit.zpa_from_freq(qfit.freq(zpa) - vdf*fs_scale)
        return vt, vgz, vqz, vdf
    
    gfit = GmonSpec(r=0.8, period=1, shift=-0.3, amp=0.005, slope=0, offset=0)
    qfit = QubitSpec(fmin=3.8, fmax=4.6, xmin=-0.1, xmax=0.7)
    gpa = 0.14
    w, delay = 10, 300

    vt, vgz, vcqz, vfs = ping_pong_z(gfit, qfit, gpa, w, 0, delay)

    fig, (ax, ax2) = plt.subplots(figsize=(8,4), ncols=2, tight_layout=True, sharey=True)
    ax.set(xlabel='Time (ns)', ylabel='DAC output (arb.)')
    ax.grid()
    # ax.step(vt, vgz,  where='post', label='gmon bias')
    # ax.step(vt, vcqz, where='post', label='qubit bias')
    ax.plot(vt, vgz , label='gmon bias')
    ax.plot(vt, vcqz, label='qubit bias')
    ax.legend()

    ax2.set(xlabel='- freq. shift')
    ax2.grid()
    ax2.plot(-vfs, vgz, '.', markersize=1)
    ax2.plot(-vfs, vcqz, '.', markersize=1)