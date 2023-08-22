"""Module containing models for fitter or lmfit."""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from lmfit import Model
from lmfit.models import (
    LinearModel, 
    LorentzianModel, 
    PolynomialModel, 
    PowerLawModel,
    QuadraticModel,
    ExpressionModel,
)

from labcodes import misc


class SineModel(Model):
    """amp * np.sin(2 * np.pi * freq * x + phase) + offset"""

    def __init__(self, **kwargs):
        def sine(x, amp=1, freq=1, phase=0, offset=0):
            return amp * np.sin(2 * np.pi * freq * x + phase) + offset
        super().__init__(sine, **kwargs)
        self.set_param_hint('period', expr=f'1/{self.prefix}freq')

    def guess(self, y:np.ndarray, x:np.ndarray):
        amp = (y.max() - y.min()) / 2
        offset = y.mean()
        freq = misc.guess_freq(x, y)
        if 'freq' in self.param_hints:
            if 'value' in self.param_hints['freq']:
                freq = self.param_hints['freq']['value']
        phase = misc.guess_phase(x, y, freq)
        return self.make_params(amp=amp, offset=offset, freq=freq, phase=phase)


class ExponentialModel(Model):
    """amp * np.exp(-x * rate) + offset"""

    def __init__(self, **kwargs):
        def exp(x, amp=1, rate=1, offset=0):
            return amp * np.exp(-x * rate) + offset
        super().__init__(exp, **kwargs)
        self.set_param_hint('tau', expr=f'1/{self.prefix}rate')

    def guess(self, y:np.ndarray, x:np.ndarray):
        amp = y[0]
        offset = y[-1]
        tau = x.mean()
        return self.make_params(amp=amp, offset=offset, rate=1/tau)


class ExpSineModel(Model):
    """amp * np.exp(-x * rate) * np.sin(2 * np.pi * freq * x + phase) + offset"""

    def __init__(self, **kwargs):
        def exp_sine(x, amp=1, rate=1, freq=1, phase=0, offset=0):
            return (amp * np.exp(-x * rate) * np.sin(2 * np.pi * freq * x + phase) + offset)
        super().__init__(exp_sine, **kwargs)
        self.set_param_hint('tau', expr=f'1/{self.prefix}rate')
        self.set_param_hint('period', expr=f'1/{self.prefix}freq')

    def guess(self, y:np.ndarray, x:np.ndarray):
        amp = (y.max() - y.min()) / 2
        offset = y.mean()
        tau = x.mean()
        freq = misc.guess_freq(x, y)
        if 'freq' in self.param_hints:
            if 'value' in self.param_hints['freq']:
                freq = self.param_hints['freq']['value']
        phase = misc.guess_phase(x, y, freq)
        return self.make_params(amp=amp, offset=offset, rate=1/tau, freq=freq, phase=phase)


class GaussianDecayModel(Model):
    """amp * np.exp(-(x * rate2)**2 - (x * rate1)) + offset"""

    def __init__(self, **kwargs):
        def gaussian_decay(x, amp=1, rate=1, offset=0):
            return amp * np.exp(-(x * rate)**2) + offset
        super().__init__(gaussian_decay, **kwargs)
        self.set_param_hint('tau1', expr=f'1/{self.prefix}rate1')
        self.set_param_hint('tau2', expr=f'1/{self.prefix}rate2')

    def guess(self, y:np.ndarray, x:np.ndarray):
        amp = y[0]
        offset = y[-1]
        tau = x.mean()
        return self.make_params(amp=amp, offset=offset, rate=1/tau)


class GaussianModel(Model):
    """amp * np.exp(-(x - center)**2 / (2 * width**2)) + offset"""

    def __init__(self, **kwargs):
        def exp(x, amp=1, center=0, width=1, offset=0):
            return amp * np.exp(-(x-center)**2/(2*width**2)) + offset
        super().__init__(exp, **kwargs)
        
    def guess(self, y:np.ndarray, x:np.ndarray):
        iy_min = np.argmin(y)
        iy_max = np.argmax(y)
        ix_max = np.argmax(x)
        ix_min = np.argmin(x)
        offset = y[0]
        width = 0.1 * (x[ix_max] - x[ix_min])
        if offset - y[iy_min] < y[iy_max] - offset:
            # If peak
            peak_i = iy_max
            center = x[peak_i]
            amp = y[peak_i] - offset  # Positive value
        else:
            # If dip
            dip_i = iy_min
            center = x[dip_i]
            amp = y[dip_i] - offset  # Negative value

        return self.make_params(
            amp=amp,
            center=center,
            width=width,
            offset=offset,
        )


class ResonatorModel_inverse(Model):

    def __init__(self, s21_inverse=True, **kwargs):
        def resonator_s21(x, f0, Qi, Qc, phi, amp=1):
            """amp * (1 - Qi * Qc^-1 / (1 + 2j * Qi * (x - f0) / f0))

            Qc is complex to take into account mismatches in the input and output 
            transmission impedances.

            Following the example provided lmfit:
            https://lmfit.github.io/lmfit-py/examples/example_complex_resonator_model.html
            """
            Qc = Qc * np.exp(1j*phi)
            return amp * (1 - (Qi * Qc**-1 / (1 + 2j * Qi * (x - f0) / f0)))

        def resonator_s21inv(x, f0=5e9, Qi=1e5, Qc=1e5, phi=0):
            """amp * (1 + Qi * Qc^-1 / (1 + 2j * Qi * (x - f0) / f0))

            Resonator fitting model fitting 1/s21 but not s21, equivelently assigning 
            more weights to those point around resonance. Hence ydata must be normalized.
            """
            Qc = Qc * np.exp(1j*phi)
            return 1 + (Qi / Qc / (1 + 2j * Qi * (x - f0) / f0))
        
        if s21_inverse:
            super().__init__(resonator_s21inv, **kwargs)
        else:
            super().__init__(resonator_s21, **kwargs)

    def guess(self, data, x):
        amp_guess = np.abs(data[0])*1.001  # assume x is sorted.
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
        return self.make_params(
            amp=amp_guess,
            f0=f0_guess,
            Qi=Qi_guess,
            Qc=Qc_guess,
            phi=0,
        )
    
    @staticmethod
    def plot(cfit, ax=None, fdata=500, annotate='', init=False):
        if ax is None:
            fig, ax = plt.subplots(tight_layout=True, figsize=(4.5,4))
        else:
            fig = ax.get_figure()

        ax.set(
            aspect='equal',
            xlabel='Re[$S_{21}^{-1}$]',
            ylabel='Im[$S_{21}^{-1}$]',
        )
        ax.grid(True)

        def plot(ax, x, y, sty='-'):
            detune = np.abs(x - cfit['f0'])
            delta = (cfit['f0'] / cfit['Qi']) / 2
            mask1 = detune <= delta
            mask2 = (detune <= delta * 10) & (np.logical_not(mask1))
            mask3 = np.logical_not(mask1 | mask2)
            ax.plot(y.real[mask1], y.imag[mask1], sty, color='C0')
            ax.plot(y.real[mask2], y.imag[mask2], sty, color='C1')
            ax.plot(y.real[mask3], y.imag[mask3], sty, color='C2')

        plot(ax, cfit.xdata, cfit.ydata, '.')
        plot(ax, *cfit.fdata(fdata), '-')
        if init:
            plot(ax, cfit.xdata, cfit.result.init_fit, '--')

        fmt = EngFormatter().format_eng
        ax.annotate(
            (f'$f_0$={fmt(cfit["f0"])},\n'
            +f'$Q_i$={fmt(cfit["Qi"])},\n'
            +f'$Q_c$={fmt(cfit["Qc"])},\n'
            +annotate),
            (0.5, 0.5),
            xycoords='axes fraction',
            ha='center',
            va='center',
        )
        return ax

    @staticmethod
    def normalize(s21):
        """Return scaled s21 with abs(off_resonance_s21) = 1."""
        amp = np.mean(np.concatenate(np.abs((s21[:9], s21[-9:]))))
        return s21 / amp

    @staticmethod
    def photon_num(f0, Qi, Qc, Pin, h=6.63e-34):
        """Returns photon number in the resonator."""
        n = Qc/(2*np.pi*f0) * (Qi/(Qi+Qc))**2 * Pin/(h*f0)
        return n


class TransmonModel(Model):
    """amp * np.exp(-(x - center)**2 / (2 * width**2)) + offset"""

    def __init__(self, **kwargs):
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

        p = self.prefix
        self.set_param_hint(f'{p}period', expr=f'2*abs({p}xmax - {p}xmin)')
        # Asymmetriy d = (Ej2 - Ej1) / (Ej2 + Ej1) = (s2 - s1) / (s2 + s1)
        self.set_param_hint(f'{p}d', expr=f'({p}fmin / {p}fmax)**2')
        # Area ratio r = s2 / s1 = (1+d)/(1-d)
        self.set_param_hint(f'{p}area_ratio', expr=f'(1+({p}fmin/{p}fmax)**2)/(1-({p}fmin/{p}fmax)**2)')

    def guess(self, data, x):
        """Estimate initial model parameter values from data."""
        imax = np.argmax(data)
        imin = np.argmin(data)
        fmax = data[imax]
        fmin = data[imin]
        xmax = x[imax]
        xmin = x[imin]
        return self.make_params(
            fmax=fmax,
            fmin=fmin,
            xmax=xmax,
            xmin=xmin,
        )
    
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
        if cfit['xmin'] < cfit['xmax']:
            ha1 = 'right'
            ha2 = 'left'
        else:
            ha1 = 'left'
            ha2 = 'right'
        ax.annotate(f'z={cfit["xmax"]:.3f}, f={cfit["fmax"]:.3f}', 
            (cfit['xmax'], cfit['fmax']),
            va='top', ha=ha1,
        )
        ax.annotate((f'z={cfit["xmin"]:.3f}, f={cfit["fmin"]:.3f},\n'
                    f'period={cfit["period"]:.3f},\n'
                    f'df={abs(cfit["fmax"] - cfit["fmin"]):.3f}.'), 
            (cfit['xmin'], cfit['fmin']),
            ha=ha2,
        )
        note_pos = (1,0) if cfit['xmin'] < cfit['xmax'] else (1,0.9)
        ax.annotate(f'R=$S_{{jj1}}/S_{{jj2}}$={cfit["area_ratio"]:.2f}', 
            note_pos, xycoords=ax.transAxes, va='bottom', ha='right')

        divider = make_axes_locatable(ax)
        ax2 = divider.append_axes('bottom', size='10%', pad=0.05, sharex=ax)

        ax2.plot(cfit.xdata, cfit.fdata() - cfit.ydata, 'x')
        ax2.axhline(0, color='C0')
        ax2.set_ylabel('residues')
        ax2.tick_params(axis='y', labelsize='small')
        ax2.set_xlabel(ax.get_xlabel())
        ax.set_xlabel('')
        return ax


class GmonModel(Model):
    """Model fitting Gmon induced tunable coupling.
    WARNING: The fit is sensitive to initial value, which must be provided by user."""
    _delta_interp = np.linspace(-2*np.pi, 2*np.pi, 10001)

    def __init__(self, with_slope=None, **kwargs):
        def gmon(x, r=0.9, amp=1, period=1, shift=0, offset=0, slope=0):
            delta_ext = 2*np.pi / period * (x - shift)
            delta = misc.inverse_interp(
                lambda x: x + np.sin(x) * r,  # RF SQUID bias, from Satzinger's thesis.
                delta_ext,
                self._delta_interp,
            )
            M = 1 / (r + 1/np.cos(delta))  # Reaches max when gpa=shift.
            return amp * M + offset + slope * x
        
        super().__init__(gmon, **kwargs)
        if not with_slope:
            self.set_param_hint(name='slope', vary=False, value=0)

        self.set_param_hint(name='r', max=1, min=0)  # r = L_linear / L_j0.
        self.set_param_hint(name='amp', min=0)
        self.set_param_hint(name='zero1', expr='(pi/2+r)/(2*pi/period) + shift')
        self.set_param_hint(name='zero2', expr='(pi*3/2-r)/(2*pi/period) + shift')
        self.set_param_hint(name='max_y_shift', expr='amp/(r-1)')

    def plot(self, cfit, ax=None, fdata=500):  # TODO: Include the slope feature.
        """Plot fit with results parameters.
        
        Args:
            cfit: fit with result.
            ax: ax to plot, if None, create a new ax.
            fdata: passed to cfit.fdata().

        Returns:
            ax with plot and annotations.
        """
        if ax is None:
            fig, ax = plt.subplots(tight_layout=True)
        else:
            fig = ax.get_figure()

        if fdata is not None:
            ax.plot(cfit.xdata, cfit.ydata, 'x')
            ax.plot(*cfit.fdata(fdata))

        gs = dict(ls='--', color='k', alpha=0.5)  # Guide line style
        ax.axhline(y=cfit['offset'], **gs)
        ax.annotate(f"y0={cfit['offset']:.3f}", 
            (ax.get_xlim()[0], cfit['offset']), ha='left')

        dip_y = cfit['offset'] + cfit['max_y_shift']
        ax.axhline(y=dip_y, **gs)
        ax.annotate(f"$\\Delta y_\\mathrm{{max}}={cfit['max_y_shift']:.4f}\\pm{cfit['max_y_shift_err']:.4f}$", 
            (ax.get_xlim()[0], dip_y), va='bottom', ha='left')

        xmin, xmax = ax.get_xlim()
        for i in np.arange(-2,3):
            dip_x = (cfit['zero1']+cfit['zero2'])/2 + i*cfit['period']
            if (dip_x > xmin) and (dip_x < xmax):
                ax.axvline(x=dip_x, **gs)
                ax.annotate(f"    x={dip_x:.3f}",  # Push space for marking dip_y.
                    (dip_x, dip_y), va='bottom', ha='left', rotation='vertical')

            shift = cfit['shift'] + i*cfit['period']
            if (shift > xmin) and (shift < xmax):
                ax.axvline(x=shift, **gs)
                ax.annotate(f"x={shift:.3f}", 
                    (shift, ax.get_ylim()[1]), va='top', ha='right', rotation='vertical')

            zero1 = cfit['zero1'] + i*cfit['period']
            if (zero1 > xmin) and (zero1 < xmax):
                ax.axvline(x=zero1, **gs)
                ax.annotate(f"x={zero1:.3f}", 
                    (zero1, cfit['offset']), va='top', ha='right', rotation='vertical')

            zero2 = cfit['zero2'] + i*cfit['period']
            if (zero2 > xmin) and (zero2 < xmax):
                ax.axvline(x=zero2, **gs)
                ax.annotate(f"x={zero2:.3f}", 
                    (zero2, cfit['offset']), va='top', ha='left', rotation='vertical')

        ax.annotate(f"$R=L_\\mathrm{{linear}}/L_{{j0}}={cfit['r']:.3f}\\pm{cfit['r_err']:.4f}$", 
            (1,0), xycoords=ax.transAxes, va='bottom', ha='right')

        return ax
