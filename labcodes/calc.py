"""Modules contains class calculating parameters of models, e.g. Transmon, Gmon, and T coupler."""

import inspect
from functools import wraps

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
from matplotlib.ticker import EngFormatter
from scipy.optimize import fsolve

from labcodes import plotter

Phi_0 = const.h / (2*const.e)  # Flux quantum.

# NOTE: Quantities here are in SI units unless noted.
# NOTE: Energy, angular velocity are converted into Hz.

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
    def Ec(self, c):
        """Ec in Hz"""
        return const.e**2 / (2*c) / const.h  # in Hz instead of J.

class Junction(Calculator):
    w = 0.4e-6
    h = 0.2e-6
    # R*S = Oxidation constant, 650 Ohm*um^2 is and emprical value.
    rs_const = 650 * 1e-6**2

    @dept
    def s(self, w, h):
        return w*h

    @dept
    def rn(self, s, rs_const):
        """Junction resistance."""
        return rs_const / (s)

    @dept
    def Lj0(self, rn):
        return 20.671 * rn/1e3 / (1.8*np.pi**2) * 1e-9  # Formula from ZYP.

    @dept
    def Ic(self, Lj0):
        return Phi_0 / (2*np.pi*Lj0)

    @dept
    def Ej(self, Ic):
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
    def E10(self, Ec, Ej):
        """E10 in Hz."""
        return np.sqrt(8*Ec*Ej) - Ec  # Hz.

    @dept
    def Em(self, m, Ec, Ej):
        """Energy of levels, m=0, 1, 2..., in Hz."""
        return m*np.sqrt(8*Ec*Ej) - Ec/12 * (6*m**2 + 6*m + 3)  # Hz.

    @dept
    def Lq(self, E10, Ec):
        return 2*Ec*const.h / (2*np.pi*E10*const.e)**2  # H.

    @dept
    def Lq_by_c(self, E10, c):
        return 1/(2*np.pi*E10)**2 / c

    def demo_Ej_vs_area(self, w=None):
        if w is None: w = np.linspace(50e-9,1e-6)
        jj = Junction(w=w)
        qb = self.new(jj=jj)

        fig, ax = plt.subplots(tight_layout=True)
        ax.set(
            title='Ej ~ jj area linearly',
            xlabel='Junc area ($um^2$)',
            ylabel='Ej (GHz)',
        )
        ax.grid(True)
        ax.plot(jj.s()/1e-12, jj.Ej()/1e9)
        def s2w(s): return s / (jj.h/1e-6)
        def w2s(w): return (jj.h/1e-6) * w
        secx = ax.secondary_xaxis('top', functions=(s2w, w2s))
        secx.set_xlabel(f'Junc width (um) (jjh={jj.h/1e-6}um)')
        # def ej2e(Ej): return qb.E10(Ej=Ej)
        # from labcodes.misc import inverse
        # e2ej = lambda E10: inverse(ej2e, E10, x0=qb.E10())
        def ej2e(Ej): return np.interp(Ej, qb.E10()/1e9, qb.Ej()/1e9)
        def e2ej(E10): return np.interp(E10, qb.Ej()/1e9, qb.E10()/1e9)
        secy = ax.secondary_yaxis('right', functions=(ej2e, e2ej))
        secy.set_ylabel(f'E10 (GHz) (Ec={qb.Ec()/1e6:.1f}MHz)')

        return ax

    @dept
    def demo_Lq_vs_Lj0(self, Lj0=None):
        if Lj0 is None: Lj0 = np.linspace(8e-9,13e-9)
        qb = self.new()
        qb.jj.Lj0 = np.linspace(8e-9,13e-9)

        fig, ax = plt.subplots()
        ax.plot(qb.jj.Lj0/1e-9, qb.jj.Lj0/1e-9, 'r-', label='$L_{j0}$')
        norm = mpl.colors.Normalize(10, 300)
        cm = plt.get_cmap('gray')
        for c in np.linspace(norm.vmin,norm.vmax,11)*1e-15:
            ax.plot(qb.jj.Lj0/1e-9, 1/(2*np.pi*qb.E10(c=c))**2 / c / 1e-9, '--', color=cm(norm(c/1e-15)))

        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cm), ax=ax, label='capacitance (fF)')
        ax.set(
            title='$L_q$ vs $L_{j0}$',
            xlabel='$L_{j0}$ (nH)',
            ylabel='$L_q=1/\\omega^2 C$ (nH)'
        )
        ax.legend()
        return ax


# if __name__ == '__main__':
#     qb = Transmon()
#     # qb.demo_ej_vs_area(w=np.linspace(50e-9,1e-6))
#     qb.demo_Lq_vs_Lj0()
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
    def delta(self, delta_ext, L_linear, Lj0):
        """Junction phase difference in presence of external bias."""
        delta = _vsolve(delta_ext, L_linear, Lj0)
        return delta

    def demo(self, delta_ext=None):
        if delta_ext is None: np.linspace(0,2*np.pi)
        squid = self.new(delta_ext=delta_ext)

        fig, ax = plt.subplots()
        r = squid.L_linear / squid.Lj0()
        ax.set(
            title=f'RF SQUID, r=$L_{{linear}}/L_{{j0}}$={r:.3f}',
            xlabel='Delta_ext (rad)',
            ylabel='Delta (rad)',
        )
        ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
        ax.xaxis.set_major_formatter(plotter.misc.multiple_formatter())
        ax.yaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
        ax.yaxis.set_major_formatter(plotter.misc.multiple_formatter())
        ax.grid()
        ax.plot(squid.delta_ext, squid.delta())

# if __name__ == '__main__':
#     squid = RF_SQUID()
#     squid.demo(delta_ext=np.linspace(0,2*np.pi))
#     plt.show()

class Gmon(RF_SQUID):
    Lg = 0.2e-9
    Lw = 0.1e-9
    delta_ext = np.pi  # The maximal coupling point.

    @dept
    def L_linear(self, Lg, Lw):  # Reloading L_linear from parent class.
        return 2*Lg + Lw

    @dept
    def off_bias(self, L_linear, Lj0):
        """Delta_ext where coupling off"""
        return np.pi/2 + (L_linear / Lj0)

    @dept
    def max_bias(self, L_linear, Lj0):
        """Delta_ext where coupling is maximal (negative)."""
        return np.pi/2 - (L_linear / Lj0)

    @dept
    def M_max(self, Lj0, Lg, Lw):
        """Maximal M (negative)."""
        return Lg**2 / (2*Lg + Lw - Lj0)

    @dept
    def M(self, Lj0, Lg, Lw, delta):
        return Lg**2 / (2*Lg + Lw + Lj0/np.cos(delta))

    @dept
    def w1_shift(self, g, Lg, L1, L2):
        return g * np.sqrt((Lg+L2) / (Lg+L1))

    @dept
    def g(self, M, L1, L2, w1, w2, Lg):
        return 0.5 * M / np.sqrt((L1+Lg)*(L2+Lg)) * np.sqrt(w1*w2)

    @dept
    def tau(self, f, M, Lq, Z):
        """1/kappa for qubit coupling to cable. f in Hz."""
        return Lq*Z / (2*np.pi*f*M)**2

    def demo(self, delta_ext=None):
        if delta_ext is None: delta_ext = np.linspace(0,2*np.pi,200)
        gmon = self.new(delta_ext=delta_ext)

        r = gmon.L_linear() / gmon.Lj0()

        fig, ax = plt.subplots()
        ax.set(
            title=f'Gmon, r=$L_{{linear}}/L_{{j0}}$={r:.3f}',
            xlabel='Delta_ext (rad)',
            ylabel='Mutual inductance L (nH)',
        )
        ax.grid()
        ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
        ax.xaxis.set_major_formatter(plotter.misc.multiple_formatter())

        ax.plot(gmon.delta(), gmon.M()/1e-9, label='vs delta')
        ax.plot(gmon.delta_ext, gmon.M()/1e-9, label='vs delta_ext')
        ax.legend()
        return ax

# if __name__ == '__main__':
#     gmon = Gmon(Lj0=0.65e-9)
#     gmon.demo(delta_ext=np.linspace(0,2*np.pi,200))
#     plt.show()

# if __name__ == '__main__':
#     gmon = Gmon()
#     factor = gmon.M_max(Lj0=0.66e-9)**2 * 36  # kappa ~= 1/36 ns-1
#     tau = factor / gmon.M_max(Lj0=0.6e-9)**2
#     print(f'tau: {tau:.2f} ns')

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
    def eta(self, c1c, c2c, c12, cc):
        """Dimensionless ratio showing indirect coupling strength comparing to direct one."""
        return (c1c*c2c) / (c12*cc)

    @dept
    def g12(self, c12, c1, c2, w1, w2):
        """Coupling by C12 only, not including the whole capacitance network."""
        return 0.5 * c12 / np.sqrt(c1*c2) * np.sqrt(w1*w2)  # by c12 only, not include C network.

    @dept
    def g_in(self, wc, w1, w2, eta, g12):
        """Indirect coupling via 010 and 111 state."""
        f_in = wc/4 * (1/(w1-wc) + 1/(w2-wc) - 1/(w1+wc) - 1/(w2+wc)) * eta
        return g12 * f_in

    @dept
    def g_di(self, eta, g12):
        """Direct coupling via capatance network."""
        return g12 * (eta + 1)

    @dept
    def g(self, g_di, g_in):
        """The tunable coupling with wc."""
        return g_di + g_in

    @dept
    def g1c(self, w1, wc, c1, cc, c1c):
        return 0.5 * c1c / np.sqrt(c1*cc) * np.sqrt(w1 * wc)

    def demo(self, wc=None):
        if wc is None: wc = np.linspace(4.3e9, 7e9)

        fig, ax = plt.subplots(tight_layout=True)
        ax.set(
            title='TCoupler',
            xlabel='Coupler freq (Hz)',
            ylabel='Qubits swap frequency (Hz)',
        )
        ax.xaxis.set_major_formatter(EngFormatter(places=1))
        ax.yaxis.set_major_formatter(EngFormatter(places=1))
        ax.grid()

        ax.plot(wc, 2*self.g(wc=wc))

        return ax


# if __name__ == '__main__':
#     tcplr = TCoupler()
#     # With default it should be 1.5 and -1.38, same as @yan_tunable_2018.
#     print('The directive coupling factor is:\n', tcplr.g_di()/tcplr.g12())
#     print('The indirective coupling factor is:\n', tcplr.g_in()/tcplr.g12())

#     # With another set of values, This plot should recovers fig.2(b) in @yan_tunable_2018.
#     tcplr = TCoupler(
#         c1=70e-15,
#         c2=72e-15,
#         cc=200e-15,
#         c1c=4e-15,
#         c2c=4.2e-15,
#         c12=0.1e-15,
#         w1=4e9,
#         w2=4e9,
#     )
#     ax = tcplr.demo(wc=np.linspace(4.3e9, 7e9))
#     plt.show()


# TODO: resonator: c, g, chi, Q, cpl_len...

class ThermalDistribution(Calculator):
    freq = 4e9
    temp = 50e-3

    @dept
    def E(self, freq):
        return const.h * freq

    @dept
    def ET(self, temp):
        return const.Boltzmann * temp

    @dept
    def boltzmann(self, E, ET):
        return 1 / np.exp(E/ET)

    @dept
    def bose(self, E, ET):
        return 1 / (np.exp(E/ET) - 1)

    @dept
    def fermi(self, E, ET):
        return 1 / (np.exp(E/ET) + 1)

    def demo(self, t=None):
        if t is None: t = np.linspace(30e-3, 80e-3)
        dist = self.new()

        fig, ax = plt.subplots()
        ax.plot(t, dist.boltzmann(temp=t), label='Boltzmann dist.')
        ax.plot(t, dist.bose(temp=t), label='Bose dist.')
        ax.plot(t, dist.fermi(temp=t), label='Fermi-Dirac dist.')
        ax.grid(True)
        ax.legend()
        ax.set(
            xlabel='Temperature (K)',
            ylabel='Thermal population',
            title=f'f={dist.freq/1e9:.2f} GHz',
        )
        return ax

# if __name__ == '__main__':
#     dist = ThermalDistribution(f=4e9)
#     dist.demo()

class Cable(Calculator):
    """Must init with Ll or Cl. The manufacturer says:

        Cl = 94 pF/m (1.2mm)
        Ll = 235 nH/m (1.2mm)

        Cl = 86.5 pF/m (2.2mm)
        Ll = 216.25 nH/m (2.2mm)
    """
    fFSR = 1e6  # Hz
    len = 100  # m

    @dept
    def vp(self, fFSR, len):
        return 2*len*fFSR

    @dept
    def Ll(self, vp, Cl):
        return 1/vp**2/Cl

    @dept
    def Cl(self, vp, Ll):
        return 1/vp**2/Ll

    @dept
    def Z(self, Ll, Cl):
        return np.sqrt(Ll/Cl)

    @dept
    def Lm(self, len, Ll):
        return len*Ll/2

    @dept
    def tau(self, g, fFSR):
        """g in Hz, returns tau in s."""
        return fFSR / (4*np.pi**2 * g**2)  # s.

    @dept 
    def g(self, tau, fFSR):
        """tau in s, returns g in Hz."""
        return np.sqrt(fFSR / (4*np.pi**2 * tau))  # Hz.

    # @dept
    # def kappa(self, g, wFSR):
    #     """Fermi's golden rule. g, wFSR in s-1, returns kappa in s-1."""
    #     # No unit conversion! The 2*pi comes from intergration of sin(x)^2/x^2 
    #     # filter function by sinusoidal drive signal (square wave also has this 
    #     # form). For detail please refer to textbook about time-depedent perturbation.
    #     return 2*np.pi * g**2 / wFSR

    def demo(self, len=None):
        if len is None: len = np.linspace(0.3,1.2)
        cb = self.new()

        fig, ax = plt.subplots()
        ax.set(
            title=f'$v_p$={cb.vp()/1e8:.3f} $\\times 10^8$ m/s',
            xlabel='length (m)',
            ylabel='fFSR (MHz)',
        )

        ax2 = ax.twinx()
        ax2.set_ylabel('Impendance ($\\Omega$)', color='C1')
        ax2.tick_params('y', labelcolor='C1')

        ax3 = ax.twinx()
        ax3.spines.right.set_position(("axes", 1.2))
        ax3.set_ylabel('Inductance (nH)', color='C2')
        ax3.tick_params('y', labelcolor='C2')

        plotter.cursor(ax, x=cb.len, y=cb.fFSR/1e6)
        cb.fFSR = cb.vp()/(2*len)
        ax.plot(len, cb.fFSR / 1e6)
        ax2.plot(len, cb.Z(len=len), color='C1')
        ax3.plot(len, cb.Lm(len=len)/1e-9, color='C2')
        return ax

# if __name__ == '__main__':
#     qb = Transmon(c=95e-15, jj=Junction(Lj0=15e-9))
#     print(f'Ec: {qb.Ec()/1e6:.1f} MHz, E10: {qb.E10()/1e9:.3f} GHz')

#     cb = Cable(fFSR=120e6, len=1, Cl=86.5e-12)
#     print(f'Ll: {cb.Ll()/1e-9:.1f} nH/m, Lm: {cb.Lm()/1e-9:.1f} nH')

#     gmon = Gmon()
#     freq_shift = gmon.w1_shift(g=cb.g(tau=22.2e-9), L1=qb.Lq(), L2=cb.Lm())
#     print(f'freq shift: {freq_shift/1e6:.2f} MHz')

#     cb.demo()
#     plt.show()

