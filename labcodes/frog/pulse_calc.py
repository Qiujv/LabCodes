# %%
import numpy as np
import matplotlib.pyplot as plt
from functools import wraps
from scipy.optimize import fsolve

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

def linear(x, slope, offset):
    return x*slope + offset

def fshift_xtalk(gpa, r, period, shift, amp, slope, **kw):
    return fshift(gpa, r, period, shift, amp) + linear(gpa, slope, offset=0.)

def _fshift(gpa, r, period, shift, amp, **kw):
    delta = junc_phase(2.*np.pi/period*(gpa-shift),r)
    M = 1. / (r + 1./np.cos(delta))
    return M*amp

def fshift(gpa, r, period, shift, amp, **kw):
    fs_0 = _fshift(0,   r, period, shift, amp)
    fs_a = _fshift(gpa, r, period, shift, amp)
    return fs_a - fs_0  # Make sure fs(gpa=0) = 0

def fshift_sqr(gpa, r, period, shift, amp, **kw):
    fs = fshift(gpa, r, period, shift, amp)
    sign = np.sign(fs)
    return sign * fs**2.

def delta_ext(delta, r):
    return delta + np.sin(delta) * r

def junc_phase(de, r):
    return inverse(lambda x: delta_ext(x, r=r), de, x0=de)

def soft_edge(t, alpha=1.):
    """erf like function, for sech shape a_out. Sattles beyond [-10, 10]."""
    return np.exp(t) / (1. + np.exp(t)) * alpha / (1. + (1.-alpha)*np.exp(t))

def soft_edges_sample(t0, t1, w, alpha=1.0, cutoff_w=10., sample_rate=1., 
                      edge='both', pwlin=True, vertical_sample=False, mute_in_mid=True):
    if alpha != 1: vertical_sample = False  # For alpha != 1, soft_edges is no longer monotonic, hence vertical sample not available.
    if vertical_sample:
        # Sample soft edge evenly on amplitude axis, reduces 3/4 points at same sample rate.
        if cutoff_w > 10.:
            print('WARNING: vertical sampling is not compatiable with cutoff_w > 10,'
                  'given cutoff_w={}.'.format(cutoff_w))
            cutoff_w = 10.
        n_pts = int(0.5*w*cutoff_w)  # For points density ~ sample_rate at region with largest slope.
        ey = np.linspace(soft_edge(-cutoff_w), soft_edge(cutoff_w), n_pts)
        ex = w * inverse(soft_edge, ey, xlim=(-cutoff_w, cutoff_w))
    else:
        # Sample soft edge evenly on time axis, requires less inverse computation.
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
                px = np.hstack([rex, rex[-1]+1., fex[0]-1., fex])
                py = np.hstack([rey, 0, 0, fey])
            else:
                px = np.hstack([rex, fex])
                py = np.hstack([rey, fey])
    elif edge == 'rise':
        px = np.hstack((rex, rex[-1]+1.))
        py = np.hstack((rey, 0))
    elif edge == 'fall':
        px = np.hstack((fex[0]-1., fex))
        py = np.hstack((0, fey))
    else:
        raise ValueError('edge should have be "both", "rise" or "fall", {} got'.format(edge))
    
    # shift, pad smaple points to produce parameter for pwlin.
    pxs = (px[:-1] + px[1:]) / 2.  # Shift points from center to left of steps.
    pys = py[1:]
    d = pxs[0] - px[0]
    pxs = np.hstack(([pxs[0]-d], pxs, [pxs[-1]+d]))  # Pad the first and last points.
    pys = np.hstack(([py[0]], pys, [py[-1]]))

    if False:
        # Check plot.
        fig, ax = plt.subplots()
        ax.set(
            title='Soft pulse sampling',
            xlabel='time (ns)',
            ylabel='voltage (arb.)',
            ylim=(-0.05,1.05),
        )
        ax.grid(True)
        ax.step(px, py, marker='x', where='mid', color='k', alpha=0.5, lw=1)  # Comparing target.
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

def transmon_fit(x, fmax, fmin, xmax, xmin):
    """Frequency of transmon, following koch_charge_2007 Eq.2.18."""
    phi = 0.5 * (x - xmax) / (xmin - xmax)  # Rescale [xmax, xmin] to [0,0.5], i.e. in Phi_0.
    d = (fmin / fmax) ** 2
    f = fmax * np.sqrt(np.abs(np.cos(np.pi*phi))
                        * np.sqrt(1. + d**2 * np.tan(np.pi*phi)**2))
    return f

def qbias_compensate_fshift(fs, fmax, fmin, xmax, xmin, idle=0):
    """Return qubit zpa s.t. f(zpa) = f(idle) - fs."""
    qkw = dict(fmin=fmin, fmax=fmax, xmin=xmin, xmax=xmax)
    f0 = transmon_fit(idle, **qkw)
    fc = f0 - fs  # Target qubit freq.
    mask = (fc < fmin) | (fc > fmax)
    if np.any(mask):
        msg = 'WARNING: Qubit should have be bias to {} is out of range [{}, {}]'
        print(msg.format(fc[mask], fmin, fmax))
    vz = inverse(lambda x: transmon_fit(x, **qkw), fc, xlim=(xmin, xmax))
    return vz  # this value include zpa_idle, but not increments for fshift only.

def ping_pong_z(  # For simulation purpose.
    gkw, qkw, 
    gpa, width, t0, delay,
    goff=0.0, gmon_r=None,
    zpa=0.0, fs_scale=1.0,
    alpha=1.0, edge='both',
    cutoff_w=10.0, sample_rate=1.0, vertical_sample=False,
):
    qkw = qkw.copy()
    gkw = gkw.copy()
    if gmon_r: gkw['r'] = gmon_r

    vt, vy = soft_edges_sample(t0, t0+delay, width, pwlin=False, vertical_sample=vertical_sample,
        sample_rate=sample_rate, edge=edge, cutoff_w=cutoff_w, alpha=alpha)
    kmin, kmax = fshift_sqr(goff, **gkw), fshift_sqr(gpa, **gkw)
    vks = vy*(kmax-kmin) + kmin

    shift = gkw['shift'] + gkw['period'] * ((goff - gkw['shift']) // gkw['period'])
    dip = shift + gkw['period']/2
    gmin, gmax = (shift, dip) if goff <= dip else (dip, shift)
    vgz = inverse(lambda x: fshift_sqr(x, **gkw), vks, xlim=(gmin,gmax))

    f0 = fshift(0, **gkw)
    vfs = fshift(vgz, **gkw)
    if fs_scale != 0:
        vcqz = qbias_compensate_fshift((vfs-f0)*fs_scale, idle=zpa, **qkw)
    else:
        vcqz = np.zeros(np.shape(vfs))

    return vt, vgz, vcqz, vfs

if __name__ == '__main__':
    gkw = dict(r=0.8, period=1.0, shift=-0.3, amp=0.005, slope=0.0)
    qkw = dict(fmin=3.8, fmax=4.6, xmin=-0.1, xmax=0.7)
    gpa = 0.14
    w, delay = 10, 300

    vt, vgz, vcqz, vfs = ping_pong_z(gkw, qkw, gpa, w, 0, delay, vertical_sample=True)

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
