"""Modules contains class calculating parameters of models, e.g. Transmon, Gmon, and T coupler."""

import attr
import typing
from functools import wraps
import inspect
import numpy as np

# A calculator class has two kind of attributes only: values and functions accepting 
# only keyword arguments.

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

@attr.s(auto_attribs=True)
class TCoupler(object):
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
        return (c1c*c2c) / (c12*cc)

    @use_attr_as_default
    def g12(self, c12, c1, c2, w1, w2, **kw):
        """Coupling by C12 only, not including the whole capacitance network."""
        return c12 / np.sqrt(c1*c2) * np.sqrt(w1*w2)  # by c12 only, not include C network.

    @use_attr_as_default
    def f_in(self, wc, w1, w2, eta, **kw):
        return wc/4 * (1/(w1-wc) + 1/(w2-wc) - 1/(w1+wc) - 1/(w2+wc)) * eta

    @use_attr_as_default
    def g_in(self, f_in, g12, **kw):
        """Indirect coupling via 010 and 111 state."""
        return 0.5 * g12 * f_in

    @use_attr_as_default
    def f_di(self, eta, **kw):
        return eta + 1

    @use_attr_as_default
    def g_di(self, f_di, g12, **kw):
        """Direct coupling via capatance network."""
        return 0.5 * g12 * f_di

    @use_attr_as_default
    def g(self, g12, f_in, f_di, **kw):
        """The tunable coupling with wc."""
        return 0.5 * g12 * (f_in + f_di)

    @use_attr_as_default
    def g1c(self, w1, wc, c1, cc, c1c, **kw):
        return 0.5 * c1c / np.sqrt(c1*cc) * np.sqrt(w1 * wc)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    tcplr = TCoupler()
    # With default it should be 1.5 and -1.38, same as @yan_tunable_2018.
    print('The directive coupling strength (dimensionless) is:\n', tcplr.f_di)
    print('The indirective coupling strength (dimensionless) is:\n', tcplr.f_in)

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
