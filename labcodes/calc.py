"""Modules contains class calculating parameters of models, e.g. Transmon, Gmon, and T coupler."""

import inspect
from functools import wraps

import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

Phi_0 = const.h / (2*const.e)  # Flux quantum.

# NOTE: Quantities here are in SI units unless noted.

def dept(f):
    """Wrap function to be a dependent on object attrbites.

    All the arguments will have their default value in object attributes, if found.
    """
    @wraps(f)
    def wrapped_f(self, *args, **user_kw):
        debug = user_kw.get('debug', False)
        kw = {}  # Function kwargs to be filled as required.
        fsig = inspect.signature(f)
        for arg_name in fsig.parameters.keys():
            # Search for required arguements one by one.
            if arg_name in user_kw:
                kw[arg_name] = user_kw[arg_name]
            else:  # Search in self attributes.
                try:
                    attr = getattr(self, arg_name)
                except AttributeError:
                    continue  # Leave blank and proceed to and TypeError: missing argument in function call.

                if callable(attr):
                    if attr == f: raise Exception('Loop evaluation!')  # TODO: Need a smarter checker.
                    if debug is True: print(arg_name)
                    kw[arg_name] = attr(**user_kw)  # Recursion here.
                else:
                    kw[arg_name] = attr

        return f(self, *args, **kw)
    return wrapped_f

class Calculator(object):
    """Calculators are bacically a dict, with properties derived from that."""
    def __init__(self, **kwargs):  # TODO: __init__(self, a=2, b=3, ...)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def new(self, **kwargs):
        kw = self.__dict__.copy()
        kw.update(kwargs)
        return self.__class__(**kw)

    @property
    def indeps(self):
        d = {}
        for k in dir(self):
            if not k.startswith('_'):
                if k != 'indeps':  # Avoid infinite recursion.
                    v = getattr(self, k, None)
                    if not callable(v):
                        if isinstance(v, Calculator):
                            d[k] = v.indeps  # Recursively solve the indeps.
                        else:
                            d[k] = v
        return d


class Capacitor(Calculator):
    c = 100e-15

    @dept  # i.e. Ec = dept(Ec)(self, **kwargs)
    def Ec(self, c, **kw):  # kw for passing arguments through.
        """Ec in Hz"""
        return const.e**2 / (2*c) / const.h  # in Hz instead of J.

class Junction(Calculator):
    w = 0.4e-6
    h = 0.2e-6
    # R*S = Oxidation constant, 650 Ohm*um^2 is and emprical value.
    rs_const = 650 * 1e-6**2

    @dept
    def s(self, w, h, **kw):
        return w*h

    @dept
    def rn(self, s, rs_const, **kw):
        """Junction resistance."""
        return rs_const / (s)

    @dept
    def Lj0(self, rn, **kw):
        return 20.671 * rn/1e3 / (1.8*np.pi**2) * 1e-9  # Formula from ZYP.

    @dept
    def Ic(self, Lj0, **kw):
        return Phi_0 / (2*np.pi*Lj0)

    @dept
    def Ej(self, Ic, **kw):
        """Ej in Hz."""
        return Ic*Phi_0 / (2*np.pi) / const.h  # Hz

# Such single chain structure is perfect because the final quantity varies
# with any of its relavant dependents. But whether there is performance overhead?

class Transmon(Capacitor):
    jj = Junction()  # Seperate junction attributes.

    @property
    def Ej(self):
        return self.jj.Ej  # Return a function! so the wrapper recursion preceeds with user_kw.

    @Ej.setter
    def Ej(self, value):
        self.jj.Ej = value

    @dept
    def E10(self, Ec, Ej, **kw):
        """E10 in Hz."""
        return np.sqrt(8*Ec*Ej) - Ec  # Hz.

    @dept
    def Em(self, m, Ec, Ej, **kw):
        """Energy of levels, m=0, 1, 2..., in Hz."""
        return m*np.sqrt(8*Ec*Ej) - Ec/12 * (6*m**2 + 6*m + 3)  # Hz


# if __name__ == '__main__':
#     jj = Junction(w=np.linspace(50e-9,1e-6))
#     qb = Transmon(jj=jj)

#     fig, ax = plt.subplots(tight_layout=True)
#     ax.set(
#         title='Ej ~ jj area linearly',
#         xlabel='Junc area ($um^2$)',
#         ylabel='Ej (GHz)',
#     )
#     ax.grid(True)
#     ax.plot(jj.s()/1e-12, jj.Ej()/1e9)
#     def s2w(s): return s / (jj.h/1e-6)
#     def w2s(w): return (jj.h/1e-6) * w
#     secx = ax.secondary_xaxis('top', functions=(s2w, w2s))
#     secx.set_xlabel(f'Junc width (um) (jjh={jj.h/1e-6}um)')
#     # def ej2e(Ej): return qb.E10(Ej=Ej)
#     # from labcodes.misc import inverse
#     # e2ej = lambda E10: inverse(ej2e, E10, x0=qb.E10())
#     def ej2e(Ej): return np.interp(Ej, qb.E10()/1e9, qb.Ej()/1e9)
#     def e2ej(E10): return np.interp(E10, qb.Ej()/1e9, qb.E10()/1e9)
#     secy = ax.secondary_yaxis('right', functions=(ej2e, e2ej))
#     secy.set_ylabel(f'E10 (GHz) (Ec={qb.Ec()/1e6:.1f}MHz)')
#     plt.show()


def _delta_ext(delta, L_linear, Lj0):
    """Relation between delta and delta_ext."""
    return delta + np.sin(delta) * (L_linear / Lj0)
def _solve(delta_ext, L_linear, Lj0):
    """Solve delta from delta_ext"""
    res = fsolve(lambda delta: delta_ext - _delta_ext(delta, L_linear, Lj0), 0)
    return res[0]
_vsolve = np.vectorize(_solve)  # expand fsolve for any arguement in np.array.

class RF_SQUID(Calculator):
    jj = Junction(w=3e-6, h=0.4e-6)
    L_linear = 0.5e-9
    delta_ext = np.pi

    @property
    def Lj0(self):
        return self.jj.Lj0

    @Lj0.setter
    def Lj0(self, value):
        self.jj.Lj0 = value

    @dept
    def delta(self, delta_ext, L_linear, Lj0, **kw):
        """Junction phase difference in presence of external bias."""
        delta = _vsolve(delta_ext, L_linear, Lj0)
        return delta

# if __name__ == '__main__':
#     from labcodes import plotter

#     squid = RF_SQUID(delta_ext=np.linspace(0,2*np.pi))
#     r = squid.L_linear / squid.Lj0()
    
#     fig, ax = plt.subplots()
#     ax.set(
#         title=f'RF SQUID, r=$L_{{linear}}/L_{{j0}}$={r:.3f}',
#         xlabel='Delta_ext (rad)',
#         ylabel='Delta (rad)',
#     )
#     ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
#     ax.xaxis.set_major_formatter(plotter.misc.multiple_formatter())
#     ax.yaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
#     ax.yaxis.set_major_formatter(plotter.misc.multiple_formatter())
#     ax.grid()
#     ax.plot(squid.delta_ext, squid.delta())
#     plt.show()

class Gmon(RF_SQUID):
    Lg = 0.2e-9
    Lw = 0.1e-9
    delta_ext = np.pi  # The maximal coupling point.

    w1 = 4e9
    w2 = 4e9
    L1 = 15e-9
    L2 = 15e-9

    @dept
    def L_linear(self, Lg, Lw, **kw):  # Reloading L_linear from parent class.
        return 2*Lg + Lw

    @dept
    def M(self, Lj0, Lg, Lw, delta, **kw):
        return Lg**2 / (2*Lg + Lw + Lj0/np.cos(delta))

    @dept
    def g(self, M, L1, L2, w1, w2, Lg, **kw):
        return 0.5 * M / np.sqrt((L1+Lg)*(L2+Lg)) * np.sqrt(w1*w2)

    @dept
    def w1_shift(self, g, Lg, L1, L2, **kw):
        return g * np.sqrt((Lg+L2) / (Lg+L1))

    @dept
    def kappa(self, g, wFSR, **kw):
        """Decay rate to multimode resonator, by Fermi's golden rule.
        In same unit as arguments."""
        # No unit conversion! The 2*pi comes from intergration of sin(x)^2/x^2 
        # filter function by sinusoidal drive signal (square wave also has this 
        # form). For detail please refer to textbook about time-depedent perturbation.
        return 2*np.pi * g**2 / wFSR

    @dept
    def tau(self, kappa):
        """Decay time (s) related to kappa (Hz)."""
        return 1/(2*np.pi*kappa)

    @dept
    def off_bias(self, L_linear, Lj0):
        """Delta_ext where coupling off"""
        return np.pi/2 + (L_linear / Lj0)

    @dept
    def max_bias(self, L_linear, Lj0):
        """Delta_ext where coupling is maximal (negative)."""
        return np.pi/2 - (L_linear / Lj0)

# if __name__ == '__main__':
#     from labcodes import plotter

#     gmon = Gmon(delta_ext=np.linspace(0,2*np.pi,200), Lj0=0.65e-9)
#     r = gmon.L_linear() / gmon.Lj0

#     fig, ax = plt.subplots()
#     ax.set(
#         title=f'Gmon, r=$L_{{linear}}/L_{{j0}}$={r:.3f}',
#         xlabel='Delta_ext (rad)',
#         ylabel='Mutual inductance L (nH)',
#     )
#     ax.grid()
#     ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
#     ax.xaxis.set_major_formatter(plotter.misc.multiple_formatter())

#     ax.plot(gmon.delta(), gmon.M()/1e-9, label='vs delta')
#     ax.plot(gmon.delta_ext, gmon.M()/1e-9, label='vs delta_ext')
#     ax.legend()
#     plt.show()

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

    @dept
    def eta(self, c1c, c2c, c12, cc, **kw):
        """Dimensionless ratio showing indirect coupling strength comparing to direct one."""
        return (c1c*c2c) / (c12*cc)

    @dept
    def g12(self, c12, c1, c2, w1, w2, **kw):
        """Coupling by C12 only, not including the whole capacitance network."""
        return 0.5 * c12 / np.sqrt(c1*c2) * np.sqrt(w1*w2)  # by c12 only, not include C network.

    @dept
    def g_in(self, wc, w1, w2, eta, g12, **kw):
        """Indirect coupling via 010 and 111 state."""
        f_in = wc/4 * (1/(w1-wc) + 1/(w2-wc) - 1/(w1+wc) - 1/(w2+wc)) * eta
        return g12 * f_in

    @dept
    def g_di(self, eta, g12, **kw):
        """Direct coupling via capatance network."""
        return g12 * (eta + 1)

    @dept
    def g(self, g_di, g_in, **kw):
        """The tunable coupling with wc."""
        return g_di + g_in

    @dept
    def g1c(self, w1, wc, c1, cc, c1c, **kw):
        return 0.5 * c1c / np.sqrt(c1*cc) * np.sqrt(w1 * wc)


# if __name__ == '__main__':
#     from matplotlib.ticker import EngFormatter

#     tcplr = TCoupler()
#     # With default it should be 1.5 and -1.38, same as @yan_tunable_2018.
#     print('The directive coupling factor is:\n', tcplr.g_di()/tcplr.g12())
#     print('The indirective coupling factor is:\n', tcplr.g_in()/tcplr.g12())

#     # With another set of values, This plot should recovers fig.2(b) in @yan_tunable_2018.
#     tcplr = TCoupler(
#         wc=np.linspace(4.3e9, 7e9),
#         c1=70e-15,
#         c2=72e-15,
#         cc=200e-15,
#         c1c=4e-15,
#         c2c=4.2e-15,
#         c12=0.1e-15,
#         w1=4e9,
#         w2=4e9,
#     )

#     fig, ax = plt.subplots(tight_layout=True)
#     ax.set(
#         title='TCoupler',
#         xlabel='Coupler freq (Hz)',
#         ylabel='Qubits swap frequency (Hz)',
#     )
#     ax.xaxis.set_major_formatter(EngFormatter(places=1))
#     ax.yaxis.set_major_formatter(EngFormatter(places=1))
#     ax.grid()
#     ax.plot(tcplr.wc, 2*tcplr.g())
#     plt.show()


# TODO: resonator: c, g, chi, Q, cpl_len...