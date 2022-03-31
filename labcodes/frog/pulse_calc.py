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

def fshift_xtalk(gpa, r, period, shift, amp, slope, **kw):
    return fshift(gpa, r, period, shift, amp) + linear(gpa, slope, offset=0.)

def fshift(gpa, r, period, shift, amp, **kw):
    delta = junc_phase(2.*np.pi/period*(gpa-shift),r)
    M = 1. / (r + 1./np.cos(delta))
    return M*amp

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
    ey = np.linspace(ymin, ymax, n_pt)  # e ~ edge.
    ex = inverse(bound(xmin, xmax)(soft_pulse), ey, (ymin+ymax)/2, **kw)

    if verbose:
        ax.plot(ex, ey, marker='x', label='rising edge')
        ax.plot(ex, soft_pulse(ex,**kw), color='k', alpha=0.5, lw=1)  # Comparing target.
        print('Space between sampling points:')
        print(np.diff(ex))  # Check sample density ~< 1 Sa/ns.

    # Composite rise and fall edge into a pulse. 'p' means 'pulse'.
    rex = ex  # Rising edge x.
    fex = t0+t1 - ex[::-1]  # Falling edge x.
    # Turn off output when edges are seperated far enough (5ns).
    if (fex[0] - rex[-1]) < 10:
        px = np.hstack((rex, fex))
        py = np.hstack((ey, ey[::-1]))
    else:
        px = np.hstack((rex, rex[-1]+5, fex[0]-5, fex))
        py = np.hstack((ey, 0, 0, ey[::-1]))
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
        vdt = np.diff(pxs)
        vt = pxs[:-1]
        vy = pys[:-1]
        return t_start, vdt, vt, vy
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

def transmon_fit(x, fmax, fmin, xmax, xmin):
    """Frequency of transmon, following koch_charge_2007 Eq.2.18."""
    phi = 0.5 * (x - xmax) / (xmin - xmax)  # Rescale [xmax, xmin] to [0,0.5], i.e. in Phi_0.
    d = (fmin / fmax) ** 2
    f = fmax * np.sqrt(np.abs(np.cos(np.pi*phi))
                        * np.sqrt(1 + d**2 * np.tan(np.pi*phi)**2))
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
    vz = inverse(bound(xmin, xmax, 10)(transmon_fit), fc, **qkw)
    return vz  # this value include zpa_idle, but not increments for fshift only.

if __name__ == '__main__':
    gkw = dict(r=0.75727153, period=1.12077479, shift=-0.42074943, amp=0.00539045, slope=0.00232655)
    qkw = dict(fmin=3.7320274, fmax=4.55496161, xmin=-0.01332578, xmax=0.70460876)
    # gpa = gkw['period']/2 + gkw['shift']  # Works well even with gpa=1e-4, gpa=0
    gpa = 0.14
    w, delay = 100, 190  # Ideally w depends on gpa.
    t0 = w
    t1 = t0 + delay

    vt, vy = soft_pulse_sample(t0, t1, w, verbose=True, pwlin=False)
    fmin, fmax = fshift(0, **gkw), fshift(gpa, **gkw)
    vf = vy*(fmax-fmin) + fmin  # Scale [0,1] to [fmin, fmax].
    vgpa = inverse(bound(0,gpa)(fshift), vf, x0=gpa/2, **gkw)
    vqpa = qbias_compensate_fshift(vf, **qkw)

    fig, (ax, ax2) = plt.subplots(figsize=(8,3), ncols=2, tight_layout=True)
    ax.set(
        title='Pulse emitting symmetric photon.',
        xlabel='Time (ns)',
        ylabel='DAC output (arb.)',
    )
    ax.grid()
    ax2.set(
        xlabel='gpa',
        ylabel='Freq shift',
    )
    ax2.grid()
    secax2 = ax2.twiny()
    secax2.set_xlabel('Time (ns)')

    ax.step(vt, vgpa, where='post', label='gmon bias')
    ax.step(vt, vqpa, where='post', label='qubit bias')
    ax.legend()
    secax2.step(vt, vf, where='post')
    # ax2.plot(vgpa, vf)
    vx = np.linspace(0,gpa,200)
    ax2.plot(vx, fshift_xtalk(vx, **gkw))

