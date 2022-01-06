"""Module containing models for fitter or lmfit."""

import operator

import numpy as np
import matplotlib.pyplot as plt
from labcodes import misc
from lmfit import CompositeModel, Model
import lmfit.models

COMMON_INIT_DOC = """
    Parameters
    ----------
    independent_vars : ['x']
        Arguments to func that are independent variables.
    prefix : str, optional
        String to prepend to parameter names, needed to add two Models that
        have parameter names in common.
    nan_policy : str, optional
        How to handle NaN and missing values in data. Must be one of:
        'raise' (default), 'propagate', or 'omit'. See Notes below.
    **kwargs : optional
        Keyword arguments to pass to :class:`Model`.

    Notes
    -----
    1. nan_policy sets what to do when a NaN or missing value is seen in the
    data. Should be one of:

        - 'raise' : Raise a ValueError (default)
        - 'propagate' : do nothing
        - 'omit' : drop missing data

    """


def update_param_vals(pars, prefix, **kwargs):
    """Update parameter values with keyword arguments."""
    for key, val in kwargs.items():
        pname = f"{prefix}{key}"
        if pname in pars:
            pars[pname].value = val
    pars.update_constraints()
    return pars


class MyModel(Model):
    def fit(self, data, params=None, weights=None, method='leastsq',
            iter_cb=None, scale_covar=True, verbose=False, fit_kws=None,
            nan_policy=None, calc_covar=True, **kwargs):
        if params is None:
            try:
                params = self.guess(data, **kwargs)  # pylint: disable=assignment-from-no-return
            except NotImplementedError:
                pass
        return super().fit(data, params, weights, method, iter_cb, scale_covar,
                            verbose, fit_kws, nan_policy, calc_covar, **kwargs)

    def __add__(self, other):
        """+"""
        return MyCompositeModel(self, other, operator.add)

    def __sub__(self, other):
        """-"""
        return MyCompositeModel(self, other, operator.sub)

    def __mul__(self, other):
        """*"""
        return MyCompositeModel(self, other, operator.mul)

    def __div__(self, other):
        """/"""
        return MyCompositeModel(self, other, operator.truediv)

    def __truediv__(self, other):
        """/"""
        return MyCompositeModel(self, other, operator.truediv)


class MyCompositeModel(MyModel, CompositeModel):
    def guess(self, data, **kwargs):
        pars = self.make_params()
        for mod in self.components:
            try:
                p = mod.guess(data, **kwargs)  # pylint: disable=assignment-from-no-return
                pars.update(p)
            except NotImplementedError:
                # print(err)
                continue
        return pars

    def set_param_hint(self, name, **kwargs):
        super(__class__, self).set_param_hint(name, **kwargs)
        for comp in self.components:  # Set param hints to right components.
            # if 'expr' in kwargs:  # BUG: if expr contains parameters from several components.
            #     para_in_expr = np.any(
            #         [name in kwargs['expr'] for name in comp.param_names]
            #     )
            # else:
            #     para_in_expr = False
            if name in comp.param_names:
                comp.set_param_hint(name, **kwargs)

    @property
    def param_names(self):
        """List of name of parameters in this model."""
        param_names = super().param_names
        param_from_hints = [n for n in self._param_names if n not in param_names]
        return param_names + param_from_hints

# Inherent models from lmfit.models, but with overridden methods of MyModel.
class LinearModel(MyModel, lmfit.models.LinearModel):
    pass

class LorentzianModel(MyModel, lmfit.models.LorentzianModel):
    pass

class PolynomialModel(MyModel, lmfit.models.PolynomialModel):
    pass

class PowerLawModel(MyModel, lmfit.models.PowerLawModel):
    pass

class QuadraticModel(MyModel, lmfit.models.QuadraticModel):
    pass

class ExpressionModel(MyModel, lmfit.models.ExpressionModel):
    pass

# Define some new models.
class AmpFeature(MyModel):
    """A constant named 'amp'.
    
    Implemented for composite model which fits un-normalized data.
    """

    def __init__(self, independent_vars=['x'], prefix='', nan_policy='raise',
                 **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})
        
        def const_amp(x, amp=1):
            try:
                return amp * np.ones(x.shape)
            except AttributeError:
                return amp

        super().__init__(const_amp, **kwargs)

    def guess(self, data, x=None, **kwargs):
        """Estimate initial model parameter values from data."""

        pars = self.make_params(
            amp=(data.max() - data.min()) / 2,
        )
        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = 'Constant (amp) feature' + COMMON_INIT_DOC


class OffsetFeature(MyModel):
    """A constant named 'offset'.
    
    Implemented for composite model which fits data with offset from 0.
    """

    def __init__(self, independent_vars=['x'], prefix='', nan_policy='raise',
                 **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})
        
        def const_offset(x, offset=1):
            try:
                return offset * np.ones(x.shape)
            except AttributeError:
                return offset

        super().__init__(const_offset, **kwargs)

    def guess(self, data, x=None, **kwargs):
        """Estimate initial model parameter values from data."""

        pars = self.make_params(
            offset=data.mean(),
        )
        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = 'Constant (offset) feature' + COMMON_INIT_DOC


class SineFeature(MyModel):
    """np.sin(2 * np.pi * freq * x + phase)"""

    def __init__(self, independent_vars=['x'], prefix='', nan_policy='raise',
                 **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})
        
        def sine(x, freq=1, phase=0):
            return np.sin(2*np.pi*freq*x + phase)

        super().__init__(sine, **kwargs)

    def guess(self, data, x=None, **kwargs):
        """Estimate initial model parameter values from data."""
        if x is not None:
            freq_guess = misc.find_freq_guess(x, data)
        else:
            freq_guess = 1  # A non-zero value.

        pars = self.make_params(
            phase=0,
            freq=freq_guess,
        )
        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = 'Sine feature' + COMMON_INIT_DOC


class SineModel(MyCompositeModel):
    """amp * np.sin(2 * np.pi * freq * x + phase) + offset"""

    def __init__(self, independent_vars=['x'], prefix='', nan_policy='raise',
                 **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})

        model = AmpFeature() * SineFeature()
        super().__init__(model, OffsetFeature(), operator.add, **kwargs)

    __init__.__doc__ = 'Sine model' + COMMON_INIT_DOC


class ExpFeature(MyModel):
    """np.exp(-x * rate)"""

    def __init__(self, independent_vars=['x'], prefix='', nan_policy='raise',
                 **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})
        
        def exp(x, rate=1):
            return np.exp(-x * rate)

        super().__init__(exp, **kwargs)
        
        self.set_param_hint(f'{self.prefix}tau', expr=f'1/{self.prefix}rate')

    def guess(self, data, x=None, **kwargs):
        """Estimate initial model parameter values from data."""

        if x is not None:
            guess_pt = np.argmin(abs(data - data[0]*0.367))
            tau_guess = x[guess_pt]
            x_max = x.max()
            if tau_guess < 0.1*x_max or tau_guess > 10*x_max:
                # Assume data has a reasonable range.
                tau_guess = 0.5*x_max
        else:
            tau_guess = 1  # A non-zero value.

        pars = self.make_params(
            rate=1/tau_guess,
        )
        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = 'Exponential feature' + COMMON_INIT_DOC


class Exp2Feature(MyModel):
    """np.exp(-(x * rate)**2)"""

    def __init__(self, independent_vars=['x'], prefix='', nan_policy='raise',
                 **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})
        
        def exp2(x, rate=1):
            return np.exp(-(x * rate) ** 2)

        super().__init__(exp2, **kwargs)
        
        self.set_param_hint(f'{self.prefix}tau', expr=f'1/{self.prefix}rate')

    def guess(self, data, x=None, **kwargs):
        """Estimate initial model parameter values from data."""

        if x is not None:
            guess_pt = np.argmin(abs(data - data[0]*0.367))
            tau_guess = x[guess_pt]
            x_max = x.max()
            if tau_guess < 0.1*x_max or tau_guess > 10*x_max:
                # Assume data has a reasonable range.
                tau_guess = 0.5*x_max
        else:
            tau_guess = 1  # A non-zero value.

        pars = self.make_params(
            rate=1/tau_guess,
        )
        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = 'Exponential feature' + COMMON_INIT_DOC


class ExponentialModel(MyCompositeModel):
    """amp * np.exp(-x * rate) + offset"""

    def __init__(self, independent_vars=['x'], prefix='', nan_policy='raise',
                 **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})
        
        model = AmpFeature() * ExpFeature()
        super().__init__(model, OffsetFeature(), operator.add, **kwargs)
        
    __init__.__doc__ = 'Exponential model' + COMMON_INIT_DOC


class ExpSineModel(MyCompositeModel):
    """amp * np.exp(-x * rate) * np.sin(2 * np.pi * freq * x + phase) + offset"""

    def __init__(self, independent_vars=['x'], prefix='', nan_policy='raise',
                 **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})

        model = AmpFeature() * ExpFeature() * SineFeature()
        super().__init__(model, OffsetFeature(), operator.add, **kwargs)

    __init__.__doc__ = 'Exponential sine model' + COMMON_INIT_DOC


class GaussianModel(MyModel):
    """amp * np.exp(-(x - center)**2 / (2 * width**2)) + offset"""

    def __init__(self, independent_vars=['x'], prefix='', nan_policy='raise',
                 **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})
        
        def exp(x, amp=1, center=0, width=1, offset=0):
            return amp * np.exp(-(x-center)**2/(2*width**2)) + offset

        super().__init__(exp, **kwargs)
        
    __init__.__doc__ = 'Gaussian model' + COMMON_INIT_DOC

    def guess(self, data, x=None, **kwargs):
        """Estimate initial model parameter values from data."""
        y = data
        if x is not None:
            iy_min = np.argmin(y)
            iy_max = np.argmax(y)
            ix_max = np.argmax(x)
            ix_min = np.argmin(x)
            offset_guess = y[0]
            width_guess = 0.1 * (x[ix_max] - x[ix_min])
            if offset_guess - y[iy_min] < y[iy_max] - offset_guess:
                # If peak
                peak_i = iy_max
                center_guess = x[peak_i]
                amp_guess = y[peak_i] - offset_guess  # Positive value
            else:
                # If dip
                dip_i = iy_min
                center_guess = x[dip_i]
                amp_guess = y[dip_i] - offset_guess  # Negative value

            pars = self.make_params(
                amp=amp_guess,
                center=center_guess,
                width=width_guess,
                offset=offset_guess,
            )
        else:
            pars = self.make_params()

        return update_param_vals(pars, self.prefix, **kwargs)


class GaussianDecayModel(MyCompositeModel):
    """amp * np.exp(-(x * rate)**2) + offset"""

    def __init__(self, independent_vars=['x'], prefix='', nan_policy='raise',
                 **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})

        model = AmpFeature() * Exp2Feature()
        super().__init__(model, OffsetFeature(), operator.add, **kwargs)

    __init__.__doc__ = 'Gaussial decay model' + COMMON_INIT_DOC


class GaussianDecayWithShiftModel(MyCompositeModel):
    """amp * np.exp(-(x * gau_rate)**2 - (x * exp_rate)) + offset"""

    def __init__(self, independent_vars=['x'], prefix='', nan_policy='raise',
                 **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})

        model = AmpFeature() * Exp2Feature(prefix='gau_') * ExpFeature(prefix='exp_')
        super().__init__(model, OffsetFeature(), operator.add, **kwargs)

    __init__.__doc__ = 'Gaussial decay with shift model' + COMMON_INIT_DOC


class ResonatorModel(MyModel):
    """amp * (1 - Qi * Qc^-1 / (1 + 2j * Qi * (x - f0) / f0))

    Qc is complex to take into account mismatches in the input and output 
    transmission impedances.

    Following the example provided lmfit:
    https://lmfit.github.io/lmfit-py/examples/example_complex_resonator_model.html
    """

    def __init__(self, independent_vars=['x'], prefix='', nan_policy='raise',
                 **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})
        
        def linear_resonator(x, f0, Qi, Qc, phi, amp=1):
            Qc = Qc * np.exp(1j*phi)
            return amp * (1 - (Qi * Qc**-1 / (1 + 2j * Qi * (x - f0) / f0)))
        
        super().__init__(linear_resonator, **kwargs)

        self.set_param_hint('Qi', min=0)  # Enforce Q is positive
        self.set_param_hint('Qc', min=0)  # Enforce Q is positive

    __init__.__doc__ = 'Resonator model' + COMMON_INIT_DOC

    def guess(self, data, x=None, **kwargs):
        verbose = kwargs.pop('verbose', None)
        pars = self.make_params()
        amp_guess = np.abs(data[0])*1.001  # assume x is sorted.
        if x is not None:
            norm_data = data / amp_guess
            argmin_s21 = np.abs(norm_data).argmin()
            xmin = x.min()
            xmax = x.max()
            f0_guess = x[argmin_s21]  # guess that the resonance is the lowest point
            Qi_min = 0.1 * (f0_guess/(xmax-xmin))  # assume the user isn't trying to fit just a small part of a resonance curve.
            delta_x = np.diff(x)  # assume f is sorted
            min_delta_x = delta_x[delta_x > 0].min()
            Qi_max = f0_guess/min_delta_x  # assume data actually samples the resonance reasonably
            Qi_guess = np.sqrt(Qi_min*Qi_max)  # geometric mean, why not?
            Qc_guess = Qi_guess/(1-np.abs(norm_data[argmin_s21]))
            pars = self.make_params(
                amp=amp_guess,
                f0=f0_guess,
                Qi=Qi_guess,
                Qc=Qc_guess,
                phi=0,
            )
        else:
            pars = self.make_params(
                amp=amp_guess,
            )
        if verbose:
            pars.pretty_print()
            
        return update_param_vals(pars, self.prefix, **kwargs)


class ResonatorModel_inverse(MyModel):
    """amp * (1 + Qi * Qc^-1 / (1 + 2j * Qi * (x - f0) / f0))

    Resonator fitting model fitting 1/s21 but not s21, equivelently assigning 
    more weights to those point around resonance. Hence ydata must be normalized.
    Normalization is done implicitly when calling .fit().
    """

    def __init__(self, independent_vars=['x'], prefix='', nan_policy='raise',
                 **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})
        
        def linear_resonator(x, f0=5e9, Qi=1e5, Qc=1e5, phi=0):
            """Returns 1/s21."""
            Qc = Qc * np.exp(1j*phi)
            return 1 + (Qi * Qc**-1 / (1 + 2j * Qi * (x - f0) / f0))
        
        super().__init__(linear_resonator, **kwargs)

        self.set_param_hint('Qi', min=0)
        self.set_param_hint('Qc', min=0)

    __init__.__doc__ = 'Resonator model' + COMMON_INIT_DOC

    def guess(self, data, x=None, **kwargs):
        verbose = kwargs.pop('verbose', None)
        pars = self.make_params()
        if x is not None:
            argmax_s21 = np.abs(data).argmax()
            xmin = x.min()
            xmax = x.max()
            f0_guess = x[argmax_s21]  # guess that the resonance is the lowest point
            Qi_min = 0.1 * (f0_guess/(xmax-xmin))  # assume the user isn't trying to fit just a small part of a resonance curve.
            delta_x = np.diff(x)  # assume f is sorted
            min_delta_x = delta_x[delta_x > 0].min()
            Qi_max = f0_guess/min_delta_x  # assume data actually samples the resonance reasonably
            Qi_guess = np.sqrt(Qi_min*Qi_max)  # geometric mean, why not?
            Qc_guess = Qi_guess/(1-np.abs(data[argmax_s21]))
            Qc_guess = np.abs(Qc_guess)
            pars = self.make_params(
                f0=f0_guess,
                Qi=Qi_guess,
                Qc=Qc_guess,
            )
        if verbose:
            pars.pretty_print()
            
        return update_param_vals(pars, self.prefix, **kwargs)
    
    def plot(self, cfit, ax=None, fdata=500, annotate=''):
        if ax is None:
            fig, ax = plt.subplots(tight_layout=True)
        else:
            fig = ax.get_figure()
            
        ax.plot(
            cfit.ydata.real, 
            cfit.ydata.imag,
            '.'
        )
        _, fdata = cfit.fdata(fdata)
        ax.plot(
            fdata.real,
            fdata.imag,
            'r-',
        )
        ax.annotate(
            f'$f_0$={cfit["f0"]:.3e},\n$Q_i$={cfit["Qi"]:.3e},\n$Q_c$={cfit["Qc"]:.3e}\n'+annotate,
            (0.5, 0.5),
            xycoords='axes fraction',
            ha='center',
            va='center',
        )
        ax.set(
            aspect='equal',
            xlabel='Re[$S_{21}^{-1}$]',
            ylabel='Im[$S_{21}^{-1}$]',
        )
        return ax

    @classmethod
    def remove_electric_delay(cls, freq, phase, plot=False):
        """Returns phases whose electric delay has been removed."""
        phase = np.unwrap(phase)
        lfit = fitter.CurveFit(
            xdata=freq,
            ydata=phase,
            model=LinearModel(),
        )
        if plot is True: lfit.plot()
        return phase - lfit.fdata()

    @classmethod
    def normalize(cls, s21):
        """Return normalized s21 with abs(background) = 1."""
        amp = np.mean(np.concatenate(np.abs((s21[:9], s21[-9:]))))
        return s21 / amp

    @classmethod
    def photon_num(cls, f0, Qi, Qc, Pin, h=6.63e-34):
        """Returns photon number in the resonator."""
        n = Qc/(2*np.pi*f0) * (Qi/(Qi+Qc))**2 * Pin/(h*f0)
        return n


class TransmonModel(MyModel):
    """amp * np.exp(-(x - center)**2 / (2 * width**2)) + offset"""

    def __init__(self, independent_vars=['x'], prefix='', nan_policy='raise',
                 **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})
        
        def transmon_freq(x, xmax=0, fmax=6e9, xmin=0.5, fmin=2e9):
            """Frequency of transmon, following koch_charge_2007 Eq.2.18.
            Paper found at https://journals.aps.org/pra/abstract/10.1103/PhysRevA.76.042319
            """
            phi = 0.5 * (x - xmax) / (xmin - xmax)  # Rescale [xmax, xmin] to [0,0.5], i.e. in Phi_0.
            d = (fmin / fmax) ** 2
            f = fmax * np.sqrt(np.abs(np.cos(np.pi*phi))
                               * np.sqrt(1 + d**2 * np.tan(np.pi*phi)**2))
            return f

        super().__init__(transmon_freq, **kwargs)

        self.set_param_hint(f'{self.prefix}period', 
            expr=f'2*abs({self.prefix}xmax - {self.prefix}xmin)')
        
    __init__.__doc__ = 'Transmon model' + COMMON_INIT_DOC

    def guess(self, data, x=None, **kwargs):
        """Estimate initial model parameter values from data."""
        imax = np.argmax(data)
        imin = np.argmin(data)
        fmax = data[imax]
        fmin = data[imin]
        if x is not None:
            xmax = x[imax]
            xmin = x[imin]
            pars = self.make_params(
                fmax=fmax,
                fmin=fmin,
                xmax=xmax,
                xmin=xmin,
            )
        else:
            pars = self.make_params(
                fmax=fmax,
                fmin=fmin,
            )

        return update_param_vals(pars, self.prefix, **kwargs)
    
    def plot(self, cfit, ax=None, fdata=500):
        """Plot fit with result parameters"""
        if ax is None:
            fig, ax = plt.subplots(tight_layout=True)
        else:
            fig = ax.get_figure()

        if fdata:
            ax.plot(cfit.xdata, cfit.ydata, 'o')
            ax.plot(*cfit.fdata(fdata))

        gs = dict(ls='--', color='k', alpha=0.5)  # Guide line style
        ax.axhline(cfit['fmax'], **gs)
        ax.axhline(cfit['fmin'], **gs)
        ax.axvline(cfit['xmin'], **gs)
        ax.axvline(cfit['xmax'], **gs)
        ax.annotate(f'z={cfit["xmax"]:.3f}, f={cfit["fmax"]:.3f}', (cfit['xmax'], cfit['fmax']))
        ax.annotate((f'z={cfit["xmin"]:.3f}, f={cfit["fmin"]:.3f},\n'
                    f'period={cfit["period"]:.3f},\n'
                    f'df={abs(cfit["fmax"] - cfit["fmin"]):.3f}.'), 
                    (cfit['xmin'], cfit['fmin']))
        return ax
