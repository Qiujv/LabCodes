"""Helper functions adapted from LabCodes by Qiujv, dated 0327."""

# %%
import numpy as np
import matplotlib.pyplot as plt
from functools import wraps
from scipy.optimize import fsolve

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

def linear(x, slope, offset):
    return x*slope + offset

def fshift_with_xtalk(gpa, r, period, shift, amp, slope):
    return fshift(gpa, r, period, shift, amp) + linear(gpa, slope, offset=0.)

def fshift(gpa, r, period, shift, amp):
    delta = junc_phase(2.*np.pi/period*(gpa-shift),r)
    M = 1. / (r + 1./np.cos(delta))
    return M*amp

def gpa_by_fshift(fs, r, period, shift, amp, x0=0):
    """Returns gpa from given freq shift.
    Search will not got beyond one period where x0 locates at."""
    kw = dict(r=r, period=period, shift=shift, amp=amp)
    interval = period / 2.
    xmin = x0 - (x0-shift)%interval  # Last mi
    xmax = xmin + interval
    return inverse(bound(xmin, xmax)(fshift), y=fs, x0=x0, **kw)

def delta_ext(delta, r):
    return delta + np.sin(delta) * r

def junc_phase(de, r):
    return inverse(delta_ext, de, x0=0., r=r)

def soft_edge(t):
    """erf like function, for sech shape a_out."""
    return np.exp(t) / (1. + np.exp(t))

def soft_pulse(t, t0, t1, w):
    """Return a pulse with soft edge and **amp = 1**.
    Pulse sattles within [t0-w,t1+w] (variation < 5e-5*amp).
    """
    w = w/10  # Scale to make pulse sattles within [t0-w,t1+w]
    v = soft_edge((t1-t)/w) - soft_edge((t0-t)/w)
    return v

def soft_pulse_sample(t0, t1, w, sample_rate=1, verbose=False, pwlin=True):
    """Sampling soft pulse uniformly on g axis.
    
    Args:
        t0, t1: position of the rising and falling edges.
        w: edge width.
        sample_rate: ROUGH control of sampling rate. The dense most place may 
            have the sample rate slightly over this.
    
    Returns:
        if pwlin:
            t_start, increments and y values for `envelope.pwlin`. 
            t_start ~= t0 - w.
        else:
            x_sample, y_sample.
    """
    if verbose:
        fig, ax = plt.subplots()
        ax.set(
            title='Soft pulse sampling',
            xlabel='time (ns)',
            ylabel='voltage (arb.)',
            ylim=(-0.05,1.05),
        )
        ax.grid(True)

    # Sampling rising edge.
    kw = dict(t0=t0, t1=t1, w=w)
    xmin = t0-w
    xmax = min((t0+t1)/2, t0+w)  # limit sampling in region where slope is large enough.
    ymax = soft_pulse(xmax, **kw)
    ymin = soft_pulse(xmin, **kw)
    n_pt = int(w/2 * sample_rate)
    y = np.linspace(ymin, ymax, n_pt)
    x = inverse(bound(xmin, xmax)(soft_pulse), y, (ymin+ymax)/2, **kw)

    if verbose:
        ax.plot(x, y, marker='x', label='rising edge')
        ax.plot(x, soft_pulse(x,**kw), color='k', alpha=0.5, lw=1)  # Comparing target.
        print('Space between sampling points:')
        print(np.diff(x))  # Check sample density ~< 1 Sa/ns.

    # Composite rise and fall edge into a pulse. 'p' means 'pulse'.
    px = np.hstack((x,t0+t1-x[::-1]))
    py = np.hstack((y,y[::-1]))
    # shift, pad smaple points to produce parameter for pwlin.
    pxs = (px[:-1] + px[1:]) / 2  # Shift points from center to left of steps.
    pys = py[1:]
    d = pxs[0] - px[0]
    pxs = np.hstack(([pxs[0]-d], pxs, [pxs[-1]+d]))  # Pad the first and last points.
    pys = np.hstack(([py[0]], pys, [py[0]]))

    if verbose:
        ax.step(px, py, marker='x', where='mid', color='k', alpha=0.5, lw=1)  # Comparing target.
        ax.step(pxs[:-1], pys[:-1], marker='.', where='post', label='DAC output')
        ax.axvline(t0, color='k', ls='--')
        ax.axvline(t1, color='k', ls='--')
        ax.legend()
        plt.show()

    if pwlin:
        t_start = pxs[0]
        increments = np.diff(pxs)
        vs = pxs[:-1]
        return t_start, increments, vs
    else:
        return pxs, pys

def _bound(f, x, xleft, xright, scale):
    # f(x) = y
    # x is scalar.
    x1, x2 = xleft, xright
    y1, y2 = f(x1), f(x2)
    if (x1 <= x) and (x <= x2):
        return f(x)
    else:
        return ((x-x1)*(y2-y1)/(x2-x1) + y1)*scale

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

def _v_sample(t0, t1, w, gkw, gpa, n_pt=None, verbose=False, pwlin=True):
    """DO NOT USE THIS!

    Another version of soft_pulse_sample, uniformly sample on voltage axis. 
    """
    if verbose:
        fig, ax = plt.subplots()
        ax.set(
            title='Soft pulse sampling',
            xlabel='time (ns)',
            ylabel='voltage (arb.)',
            ylim=(-0.05,1.05),
        )
        ax.grid(True)

    if n_pt is None: n_pt = int(w/2)
    # Sampling rising edge.
    f0, fpa = fshift(0, **gkw), fshift(gpa, **gkw)
    def y2f(y):
        return y*(fpa-f0) + f0
    kw = dict(t0=t0, t1=t1, w=w)
    tmin = t0-w
    tmax = min((t0+t1)/2, t0+w)  # limit sampling in region where slope is large enough.
    ymin, ymax = [soft_pulse(x, **kw) for x in (tmin, tmax)]
    fmin, fmax = [y2f(y) for y in (ymin, ymax)]
    vmin, vmax = [gpa_by_fshift(f, x0=0.2, **gkw) for f in (fmin, fmax)]
    # TODO: Continue here!
    ev = np.linspace(vmin, vmax, n_pt)
    ef = fshift(ev, **gkw)
    ey = inverse(y2f, ef, 0.5)
    et = inverse(bound(tmin, tmax)(soft_pulse), ey, (ymin+ymax)/2, **kw)

    if verbose:
        ax.plot(et, ey, marker='x', label='rising edge')
        ax.plot(et, soft_pulse(et,**kw), color='y', alpha=0.3, lw=5)  # Comparing target.
        print('Space between sampling points:')
        print(np.diff(et))  # Check sample density ~< 1 Sa/ns.

    # Composite rise and fall edge into a pulse. 'p' means 'pulse'.
    px = np.hstack((et,t0+t1-et[::-1]))
    py = np.hstack((ey,ey[::-1]))
    # shift, pad smaple points to produce parameter for pwlin.
    pxs = (px[:-1] + px[1:]) / 2  # Shift points from center to left of steps.
    pys = py[1:]
    d = pxs[0] - px[0]
    pxs = np.hstack(([pxs[0]-d], pxs, [pxs[-1]+d]))  # Pad the first and last points.
    pys = np.hstack(([py[0]], pys, [py[0]]))

    if verbose:
        ax.step(px, py, marker='x', where='mid', color='y', alpha=0.3, lw=5)  # Comparing target.
        ax.step(pxs[:-1], pys[:-1], marker='.', where='post', label='DAC output')
        ax.axvline(t0, color='k', ls='--')
        ax.axvline(t1, color='k', ls='--')
        ax.legend()
        plt.show()

    if pwlin:
        t_start = pxs[0]
        increments = np.diff(pxs)
        vs = pxs[:-1]
        return t_start, increments, vs
    else:
        return pxs, pys


if __name__ == '__main__':
    gpa = 0.5
    t0, t1, w = 25, 75, 20  # Ideally w depends on gpa.
    gkw = dict(r=0.8, period=1., shift=0., amp=0.006)

    vt, vy = soft_pulse_sample(t0, t1, w, verbose=True, pwlin=False)
    fmin, fmax = fshift(0, **gkw), fshift(gpa, **gkw)
    vf = vy*(fmax-fmin) + fmin  # Scale [0,1] to [fmin, fmax].
    va = gpa_by_fshift(vf, x0=0.3, **gkw)

    fig, (ax, ax2) = plt.subplots(figsize=(8,3), ncols=2, tight_layout=True)
    ax.set(
        title='Pulse emitting symmetric photon.',
        xlabel='Time (ns)',
        ylabel='DAC output (arb.)',
    )
    ax.grid()
    secax = ax.twinx()
    secax.set_ylabel('Freq shift', color='C1')
    ax2.set(
        title='The gmon',
        xlabel='gpa',
        ylabel='Freq shift',
    )
    ax2.grid()

    ax.step(vt, va, where='post')
    secax.step(vt, vf, where='post', color='C1')
    ax2.plot(va, vf, color='C1')
# %%
