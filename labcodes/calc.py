"""Modules contains class calculating parameters of models, e.g. Transmon, Gmon, and T coupler."""

import inspect
from functools import wraps

import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

Phi_0 = const.h / (2*const.e)  # Flux quantum.

# NOTE: Quantities here are in SI units unless noted.

def use_attr_as_default(f):
    @wraps(f)
    def wrapped_f(self, **kwargs):  # NOTE: Does not accept positonal arguments!!.
        # Get required kwargs from object attributes.
        kw = {}
        fsig = inspect.signature(f)
        for arg_name in fsig.parameters.keys():
            if arg_name in kwargs:
                continue  # Bypass those given kwargs.
            try:
                attr = getattr(self, arg_name)  # Can be value or function.
            except AttributeError:
                continue

            if callable(attr):
                if attr == f: raise Exception('Loop evaluation!')  # TODO: Break a(b(a())) look then!
                if kwargs.get('debug'): print(arg_name)  # Call f with debug=True to enable this.
                kw[arg_name] = attr(**kwargs)
            else:
                kw[arg_name] = attr

        # Pad kwargs with object attributes as default.
        kw.update(kwargs)
        return f(self, **kw)
    return wrapped_f

class Calculator(object):
    """Calculators are bacically a dict, with properties derived from that."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class Capacitor(Calculator):
    c = 100e-15

    @use_attr_as_default  # Equivelant to Ec = use_attr_as_default(Ec)(self, **kwargs)
    def Ec(self, c, **kw):  # **kw is there for passing arguments through.
        return const.e**2 / (2*c) / const.h  # in Hz but not J.

class Junction(Calculator):
    w = 0.4e-6
    h = 0.2e-6
    # R*S = Oxidation constant, 650 Ohm*um^2 is and emprical value.
    rs_const = 650 * 1e-6**2

    @use_attr_as_default
    def s(self, w, h, **kw):
        return w*h

    @use_attr_as_default
    def rn(self, w, h, rs_const, **kw):
        """Junction resistance."""
        return rs_const / (w*h)

    @use_attr_as_default
    def Lj0(self, rn, **kw):
        return 20.671 * rn/1e3 / (1.8*np.pi**2) * 1e-9  # Formula from ZYP.

    @use_attr_as_default
    def Ic(self, Lj0, **kw):
        return Phi_0 / (2*np.pi*Lj0)

    @use_attr_as_default
    def Ej(self, Ic, **kw):
        return Ic*Phi_0 / (2*np.pi) / const.h  # TODO: use Lj0 to compute it.

class RF_SQUID(Junction):
    L_linear = 0.5e-9
    delta_ext = np.pi

    @use_attr_as_default
    def delta(self, delta_ext, L_linear, Lj0, **kw):
        """Junction phase difference in presence of external bias."""
        def _delta_ext(delta, L_linear, Lj0):
            return delta + np.sin(delta) * (L_linear / Lj0)
        # fsolve does not works with np.array.
        if isinstance(delta_ext, np.ndarray):
            # Solve the values one by one instead of a high-dimensional system (it is decoupled).
            delta = [fsolve(lambda d: de - _delta_ext(d, L_linear, Lj0), 0)[0] 
                    for de in delta_ext.ravel()]
            delta = np.array(delta).reshape(delta_ext.shape)
        else:
            delta = fsolve(lambda d: delta_ext - _delta_ext(d, L_linear, Lj0), 0)[0]
        return delta

class Transmon(Junction, Capacitor):
    @use_attr_as_default
    def E10(self, Ec, Ej, **kw):
        return np.sqrt(8*Ec*Ej) - Ec

    @use_attr_as_default
    def Em(self, m, Ec, Ej, **kw):
        """Energy of levels, m=0, 1, 2..."""
        return m*np.sqrt(8*Ec*Ej) - Ec/12 * (6*m**2 + 6*m + 3)

class Gmon(RF_SQUID):
    Lg = 0.2e-9
    Lw = 0.1e-9
    delta_ext = np.pi  # The maximal coupling point.

    w1 = 4e9
    w2 = 4e9
    L1 = 15e-9
    L2 = 15e-9

    @use_attr_as_default
    def L_linear(self, Lg, Lw, **kw):  # Reloading L_linear from parent class.
        return 2*Lg + Lw

    @use_attr_as_default
    def M(self, Lj0, Lg, Lw, delta, **kw):
        return Lg**2 / (2*Lg + Lw + Lj0/np.cos(delta))

    @use_attr_as_default
    def g(self, M, L1, L2, w1, w2, Lg, **kw):
        return 0.5 * M / np.sqrt((L1+Lg)*(L2+Lg)) * np.sqrt(w1*w2)

    @use_attr_as_default
    def w1_shift(self, g, Lg, L1, L2, **kw):
        return g * np.sqrt((Lg+L2) / (Lg+L1))

    @use_attr_as_default
    def kappa(self, g, wFSR, **kw):
        """Decay rate to multimode resonator, by Fermi's golden rule.
        In same unit as arguments."""
        # No unit conversion! The 2*pi comes from intergration of sin(x)^2/x^2 
        # filter function by sinusoidal drive signal (square wave also has this 
        # form). For detail please refer to textbook about time-depedent perturbation.
        return 2*np.pi * g**2 / wFSR

    @use_attr_as_default
    def off_bias(self, L_linear, Lj0):
        """Bias point where coupling off"""
        return np.pi/2 + (L_linear / Lj0)

    @use_attr_as_default
    def max_bias(self, L_linear, Lj0):
        """Bias point where coupling is maximal (negative)."""
        return np.pi/2 - (L_linear / Lj0)

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    squid = RF_SQUID()
    gmon = Gmon()
    delta_ext = np.linspace(0,2*np.pi)
    delta = gmon.delta(delta_ext=delta_ext, Lj0=0.6e-9)
    fig, (ax, ax2) = plt.subplots(figsize=(10,4), ncols=2)
    ax.plot(delta_ext / np.pi, delta / np.pi)
    ax2.plot(delta_ext / np.pi, gmon.M(delta=delta_ext)/1e-9)
    ax2.plot(delta_ext / np.pi, gmon.M(delta=delta)/1e-9)

class TCoupler(Calculator):
    wc = 5e9
    w1 = 4e9
    w2 = 4e9

    c1 = 100e-15
    c2 = 100e-15
    cc = 100e-15
    c1c = 1e-15
    c2c = 1e-15
    c12 = 0.02e-15

    @use_attr_as_default
    def eta(self, c1c, c2c, c12, cc, **kw):
        """Dimensionless ratio showing indirect coupling strength comparing to direct one."""
        return (c1c*c2c) / (c12*cc)

    @use_attr_as_default
    def g12(self, c12, c1, c2, w1, w2, **kw):
        """Coupling by C12 only, not including the whole capacitance network."""
        return 0.5 * c12 / np.sqrt(c1*c2) * np.sqrt(w1*w2)  # by c12 only, not include C network.

    @use_attr_as_default
    def f_in(self, wc, w1, w2, eta, **kw):
        return wc/4 * (1/(w1-wc) + 1/(w2-wc) - 1/(w1+wc) - 1/(w2+wc)) * eta

    @use_attr_as_default
    def g_in(self, f_in, g12, **kw):
        """Indirect coupling via 010 and 111 state."""
        return g12 * f_in

    @use_attr_as_default
    def f_di(self, eta, **kw):
        return eta + 1

    @use_attr_as_default
    def g_di(self, f_di, g12, **kw):
        """Direct coupling via capatance network."""
        return g12 * f_di

    @use_attr_as_default
    def g(self, g12, f_in, f_di, **kw):
        """The tunable coupling with wc."""
        return g12 * (f_in + f_di)

    @use_attr_as_default
    def g1c(self, w1, wc, c1, cc, c1c, **kw):
        return 0.5 * c1c / np.sqrt(c1*cc) * np.sqrt(w1 * wc)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    tcplr = TCoupler()
    # With default it should be 1.5 and -1.38, same as @yan_tunable_2018.
    print('The directive coupling factor is:\n', tcplr.f_di())
    print('The indirective coupling factor is:\n', tcplr.f_in())

    # With another set of values, This plot should recovers fig.2(b) in @yan_tunable_2018.
    wc = np.linspace(4.3e9, 7e9)
    fig, ax = plt.subplots()
    ax.plot(
        wc/1e9,
        2*tcplr.g(
            wc=wc,
            c1=70e-15,
            c2=72e-15,
            cc=200e-15,
            c1c=4e-15,
            c2c=4.2e-15,
            c12=0.1e-15,
            w1=4e9,
            w2=4e9,
        )/1e6,
    )
