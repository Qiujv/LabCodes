"""Modules contains class calculating parameters of models, e.g. Transmon, Gmon, and T coupler."""

import inspect
import typing
from functools import wraps

import attr
import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt

Phi_0 = const.h / (2*const.e)  # Flux quantum.

# A calculator class has two kind of attributes only: values and functions accepting 
# only keyword arguments.
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
    def plot(self, yname, ax=None, plot_kw={}, **kwargs):
        """Quick plot quantities.
        
        Args:
            yname: str, name of function to call.
            **kwargs: passed to the function.
            ax: axes to plot on.
            plot_kw: passed to ax.plot.

        Returns:
            The axes.
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        xname, xval = [(k,v) for k, v in kwargs.items() 
                        if isinstance(v, np.ndarray)][0]

        ax.plot(
            xval,
            getattr(self, yname)(**kwargs),
            **plot_kw
        )
        ax.set(
            xlabel=xname,
            ylabel=yname,
        )
        return ax


@attr.s(auto_attribs=True)
class Capacitor(Calculator):
    c: float = 100e-15

    @use_attr_as_default
    def Ec(self, c, **kw):
        return const.e**2 / (2*c) / const.h  # in Hz but not J.

@attr.s(auto_attribs=True)
class Junction(Calculator):
    w: float = 0.4e-6
    h: float = 0.2e-6
    # R*S = Oxidation constant, 650 Ohm*um^2 is and emprical value.
    rs_const: float = 650 * 1e-6**2

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

@attr.s(auto_attribs=True)
class Transmon(Junction, Capacitor):
    @use_attr_as_default
    def E10(self, Ec, Ej, **kw):
        return np.sqrt(8*Ec*Ej) - Ec

    @use_attr_as_default
    def Em(self, m, Ec, Ej, **kw):
        """Energy of levels, m=0, 1, 2..."""
        return m*np.sqrt(8*Ec*Ej) - Ec/12 * (6*m**2 + 6*m + 3)

@attr.s(auto_attribs=True)
class Gmon(Junction):
    Lg: float = 0.2e-9
    Lw: float = 0.1e-9
    delta: float = np.pi  # The maximal coupling point.

    w1: float = 4e9
    w2: float = 4e9
    L1: float = 15e-9
    L2: float = 15e-9

    @use_attr_as_default
    def L_linear(self, Lg, Lw, **kw):
        return 2*Lg + Lw
    
    @use_attr_as_default
    def r(self, Lj0, L_linear, **kw):
        return L_linear / Lj0

    @use_attr_as_default
    def M(self, Lj0, L_linear, Lg, delta, **kw):
        return Lg**2 / (L_linear + Lj0/np.cos(delta))

    @use_attr_as_default
    def g(self, M, L1, L2, w1, w2, **kw):
        return 0.5 * M / np.sqrt(L1*L2) * np.sqrt(w1*w2)

    @use_attr_as_default
    def w1_shift(self, g, Lg, L1, L2, **kw):
        return g * np.sqrt((Lg+L2) / (Lg+L1))

    @use_attr_as_default
    def kappa(self, g, wFSR):
        """Decay rate to multimode resonator, by Fermi's golden rule."""
        # No unit conversion! The 2*pi comes from intergration of sin(x)^2/x^2 
        # filter function by sinusoidal drive signal (square wave also has this 
        # form). For detail please refer to textbook about time-depedent perturbation.
        return 2*np.pi * g**2 / wFSR

@attr.s(auto_attribs=True)
class TCoupler(Calculator):
    wc: float = 5e9
    w1: float = 4e9
    w2: float = 4e9

    c1: float = 100e-15
    c2: float = 100e-15
    cc: float = 100e-15
    c1c: float = 1e-15
    c2c: float = 1e-15
    c12: float = 0.02e-15

    @use_attr_as_default  # Equivelant to eta = use_attr_as_default(ete)(self, **kwargs)
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
    print('The directive coupling strength (dimensionless) is:\n', tcplr.f_di())
    print('The indirective coupling strength (dimensionless) is:\n', tcplr.f_in())

    # With another set of values, This plot should recovers fig.2(b) in @yan_tunable_2018.
    wc = np.linspace(4.3e9, 7e9)
    plt.plot(
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
