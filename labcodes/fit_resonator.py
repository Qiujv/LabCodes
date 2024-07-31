import logging

import lmfit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.constants as const

from labcodes import misc, plotter
from labcodes.peak_find import PeakFinder

logger = logging.getLogger(__name__)


def s21m1(x, Qi=1e6, Qc=1e3, f0=4, phi=0, alpha=-50, phiv=10, phi0=0, amp=1):
    """Normalized s21^-1 for linear resonator.

    Original paper see https://doi.org/10.1063/1.3693409
    Improved by Xiayu Linpeng.
    """
    Qc = Qc * np.exp(1j * phi)
    s21m1 = 1 + (Qi / Qc / (1 + 2j * Qi * (x - f0) / f0))
    bg_amp = amp / (1 + alpha * (x - f0) / f0)
    bg_phase = np.exp(1j * (phiv * (x - f0) + phi0))
    return bg_amp * s21m1 * bg_phase


model = lmfit.Model(s21m1)


def n_photon(f0_Hz, Qi, Qc, p_dBm):
    """Returns photon number in the resonator.

    See https://doi.org/10.1063/1.4919761
    """
    p_watt = 10 ** (p_dBm / 10) / 1000
    wr = 2 * np.pi * f0_Hz
    n = 2 / (const.hbar * wr**2) * Qc * (Qi / (Qi + Qc)) ** 2 * p_watt
    return n


class FitResonator:
    """Fit the s21 data of linear resonator side loaded to feedline.

    Adapted from:
    1. https://lmfit.github.io/lmfit-py/examples/example_complex_resonator_model.html
    2. programs by Xiayu Linpeng.
    """

    def __init__(
        self,
        freq: np.ndarray = None,
        s21_dB: np.ndarray = None,
        s21_rad: np.ndarray = None,
        df: pd.DataFrame = None,
        hold: bool = False,
        **fit_kws,
    ):
        """Prepare data, guess initial parameters, then fit.
        
        Usage:

        - To skip data preparation, provide `df`.

        - To manually set initial guess, pass values like `fit(Qi=1e6)`.

        - One can always modify `df`, `init_params` then call `fit()` again to custom the fit.
        """
        if df is None:
            self.df = self.prepare_data(freq, s21_dB, s21_rad)
        else:
            self.df = df.copy()[['freq', 's21_dB', 's21_rad']]

        self.init_params = None
        self.init_params = self.guess_params()

        self.result: lmfit.minimizer.MinimizerResult = None
        if not hold:
            try:
                self.fit(**fit_kws)
            except:
                logger.exception("Error in fitting.")

    @staticmethod
    def prepare_data(freq: np.ndarray, s21_dB: np.ndarray, s21_rad: np.ndarray):
        freq = np.asarray(freq)
        s21_dB = np.asarray(s21_dB)
        s21_rad = np.asarray(s21_rad)

        n_pts_bg = freq.size // 10
        s21_rad = np.unwrap(s21_rad)
        s21_dB = remove_background(s21_dB, freq, fit_mask=n_pts_bg, offset=0)
        s21_rad = remove_background(s21_rad, freq, fit_mask=slice(0, n_pts_bg), 
                                    offset=0)
        s21_rad = s21_rad - s21_rad[np.argmin(s21_dB)]  # Move phase of dip to 0.

        return pd.DataFrame(dict(freq=freq, s21_dB=s21_dB, s21_rad=s21_rad))
    
    @property
    def s21_cplx(self) -> np.ndarray:
        return self.df.eval('10 ** (s21_dB / 20) * exp(1j * s21_rad)').values

    def fit(self, **fit_kws) -> lmfit.minimizer.MinimizerResult:
        """fit(Qi=1e6) to overwrite initial guess.
        
        Defaultly uses `weights=np.abs(s21_cplx)`. Pass `weights=None` to disable any weights.
        """
        freq = self.df.freq.values
        s21_cplx = self.s21_cplx
        s21m1_cplx = 1 / s21_cplx

        if 'weights' not in fit_kws:
            # Lower weight around dip, for improved robustness with noisy data.
            # pass weights=None to disable any weights.
            fit_kws['weights'] = np.abs(s21_cplx)

        self.result = model.fit(
            s21m1_cplx,
            x=freq,
            params=self.init_params,
            method="Powell",
            **fit_kws,
        )
        return self.result

    def guess_params(self):
        freq = self.df.freq.values
        s21_dB = self.df.s21_dB.values
        s21_cplx = self.s21_cplx
        s21m1_cplx = 1 / s21_cplx

        idip = np.argmin(s21_dB)
        fr = freq[idip]

        pf = PeakFinder(freq, -np.abs(s21_cplx))
        Qc = abs(pf['center'] / pf['hwhm']) / 2

        pf = PeakFinder(freq, np.abs(s21m1_cplx))
        Qi = abs(pf['center'] / pf['hwhm'])
        if np.sum(np.real(s21m1_cplx)) < 1:
            Qi = -Qi

        freq_span = freq[-1] - freq[0]
        alpha = fr * (abs(s21_cplx[-1]) - abs(s21_cplx[0])) / freq_span
        phiv = np.angle(s21m1_cplx[-1] / s21m1_cplx[0]) / freq_span

        params = lmfit.Parameters()
        params.set(Qi=Qi, Qc=Qc, f0=fr, phi=0, alpha=alpha, phiv=phiv, phi0=0, amp=1)
        params.set(phiv=dict(min=-np.pi / freq_span, max=np.pi / freq_span))
        return params

    def __getitem__(self, key: str):
        if self.result is not None:
            if key == "chi":
                return np.sqrt(self.result.chisqr)
            elif key.endswith("_err"):
                return self.result.params[key[:-4]].stderr
            else:
                return self.result.params[key].value
        else:
            return self.init_params[key].value

    def n_photon(self, p_dBm, f0_Hz=None, Qi=None, Qc=None):
        """Returns photon number in the resonator.

        See https://doi.org/10.1063/1.4919761
        """
        if f0_Hz is None:
            f0_Hz = self["f0"]
        if Qi is None:
            Qi = self["Qi"]
        if Qc is None:
            Qc = self["Qc"]
        return n_photon(f0_Hz, Qi, Qc, p_dBm)

    def plot_cplx(
        self,
        ax: plt.Axes = None,
        freq: np.ndarray = None,
        plot_fit: bool = True,
        plot_init: bool = False,
    ):
        if ax is None:
            _, ax = plt.subplots(figsize=(3, 3), layout="compressed")
            setup_ax_cplx(ax)
        if freq is None:
            freq = self.df.freq.values

        f0 = self["f0"]
        Qi = self["Qi"]
        Qc = self["Qc"]
        dfi = abs(f0 / Qi / 2)
        dfc = abs(f0 / Qc / 2)

        def plot(ax: plt.Axes, x: np.ndarray, y: np.ndarray, sty="-", **kw):
            dx = np.abs(x - f0)
            mask1 = dx <= dfi
            mask3 = dx >= dfc
            mask2 = np.logical_not(mask1 | mask3)
            ax.plot(y.real[mask1], y.imag[mask1], sty, color="C0", **kw)
            ax.plot(y.real[mask2], y.imag[mask2], sty, color="C1", **kw)
            ax.plot(y.real[mask3], y.imag[mask3], sty, color="C2", **kw)

        plot(ax, self.df.freq.values, 1 / self.s21_cplx, ".", alpha=0.5)

        if (self.result is not None) and plot_fit:
            plot(ax, freq, self.result.eval(x=freq), "-")
            ax.annotate(
                (f"$f_0={f0:.4g}$\n"
                 f"$Q_i={Qi:+.4g}$\n"
                 f"$Q_c={Qc:+.4g}$"),
                (0.5, 0.5),
                xycoords="axes fraction",
                ha="center",
                va="center",
            )

        if plot_init:
            init_param = {k: p.init_value for k, p in self.result.params.items()}
            plot(ax, freq, model.eval(x=freq, **init_param), "--")

        return ax

    def plot_s21(
        self,
        axs: tuple[plt.Axes, plt.Axes] = None,
        freq: np.ndarray = None,
        plot_fit: bool = True,
        plot_init: bool = False,
    ):
        if axs is None:
            _, (ax, ax2) = plt.subplots(figsize=(6, 2), ncols=2)
            setup_ax_abs(ax)
            setup_ax_deg(ax2)
        else:
            ax, ax2 = axs

        if freq is None:
            freq = self.df.freq.values

        f0 = self["f0"]
        Qi = self["Qi"]
        Qc = self["Qc"]
        dfi = abs(f0 / Qi / 2)
        dfc = abs(f0 / Qc / 2)

        def plot(ax: plt.Axes, x: np.ndarray, y: np.ndarray, sty="-"):
            dx = np.abs(x - f0)
            mask1 = dx <= dfi
            mask3 = dx >= dfc
            mask2 = np.logical_not(mask1 | mask3)
            mask2a = mask2 & (x < f0)
            mask2b = mask2 & (x >= f0)
            mask3a = mask3 & (x < f0)
            mask3b = mask3 & (x >= f0)
            ax.plot(x[mask1], y[mask1], sty, color="C0")
            ax.plot(x[mask2a], y[mask2a], sty, color="C1")
            ax.plot(x[mask2b], y[mask2b], sty, color="C1")
            ax.plot(x[mask3a], y[mask3a], sty, color="C2")
            ax.plot(x[mask3b], y[mask3b], sty, color="C2")

        plot(ax, self.df.freq.values, self.df.s21_dB.values, ".")
        plot(ax2, self.df.freq.values, np.rad2deg(self.df.s21_rad.values), ".")

        if (self.result is not None) and plot_fit:
            s21_cplx = 1 / self.result.eval(x=freq)
            ax.plot(freq, 20 * np.log10(np.abs(s21_cplx)), "k-")
            ax2.plot(freq, np.rad2deg(np.unwrap(np.angle(s21_cplx))), "k-")

        if plot_init:
            init_param = {k: p.init_value for k, p in self.result.params.items()}
            s21_cplx = 1 / model.eval(x=freq, **init_param)
            ax.plot(freq, 20 * np.log10(np.abs(s21_cplx)), "--", color="gray")
            ax2.plot(freq, np.rad2deg(np.unwrap(np.angle(s21_cplx))), "--", color="gray")

        return ax, ax2

    def plot(
        self,
        axs: tuple[plt.Axes, plt.Axes, plt.Axes] = None,
        freq: np.ndarray = None,
        plot_fit: bool = True,
        plot_init: bool = False,
    ):
        if axs is None:
            _, (ax, ax2, ax3) = plt.subplots(
                figsize=(8, 3), ncols=3, layout="constrained"
            )
            setup_ax_abs(ax)
            setup_ax_deg(ax2)
            setup_ax_cplx(ax3)
        else:
            ax, ax2, ax3 = axs

        self.plot_cplx(ax=ax3, freq=freq, plot_fit=plot_fit, plot_init=plot_init)
        self.plot_s21(axs=(ax, ax2), freq=freq, plot_fit=plot_fit, plot_init=plot_init)
        return ax, ax2, ax3


def setup_ax_abs(ax: plt.Axes):
    ax.set_xlabel("freq")
    ax.set_title("s21_dB")


def setup_ax_rad(ax: plt.Axes):
    ax.set_xlabel("freq")
    ax.set_title("s21_rad")
    # ax.set_ylim(-3.2, 3.2)
    ax.yaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
    ax.yaxis.set_major_formatter(plotter.misc.multiple_formatter(2, np.pi))


def setup_ax_deg(ax: plt.Axes):
    ax.set_xlabel("freq")
    ax.set_title("s21_deg")


def setup_ax_cplx(ax: plt.Axes):
    ax.set_xlabel("Re[$S_{21}^{-1}$]")
    ax.set_ylabel("Im[$S_{21}^{-1}$]")
    ax.set_aspect("equal")
    ax.axhline(y=0, color="gray", alpha=0.5, zorder=1)
    ax.axvline(x=1, color="gray", alpha=0.5, zorder=1)


def remove_background(
    y: np.ndarray,
    x: np.ndarray = None,
    fit_mask: int | slice | np.ndarray = None,
    offset: float = None,
    plot: bool = False,
):
    """Remove linear background from data.

    Args:
        fit_mask: mask of data to use for background estimation.
            if int, use the first and last fit_mask points.
        offset: offset after background removal.
            if None, use y[0].

    Examples:
    >>> np.round(remove_background([0,3,1], fit_mask=1), decimals=2)
    array([0. , 2.5, 0. ])
    """
    if x is None:
        x = np.arange(len(y))
    y = np.asarray(y)
    x = np.asarray(x)

    if fit_mask is None:
        x_to_fit, y_to_fit = x, y
    elif isinstance(fit_mask, int):
        x_to_fit = np.r_[x[:fit_mask], x[-fit_mask:]]
        y_to_fit = np.r_[y[:fit_mask], y[-fit_mask:]]
    else:
        x_to_fit, y_to_fit = x[fit_mask], y[fit_mask]

    if offset is None:
        offset = y[0]

    bg_params = np.polyfit(x_to_fit, y_to_fit, 1)  # Linear fit.
    y_fit = np.polyval(bg_params, x)
    new_y = y - y_fit + offset
    if plot:
        _, ax = plt.subplots()
        ax.plot(x, y, ".-", label="raw")
        ax.plot(x, y_fit, label="bg")
        ax.plot(x, new_y, label="new")
        ax.legend()
        plt.show()
    return new_y



if __name__ == "__main__":
    import doctest
    doctest.testmod()

    freq = misc.segments(
        misc.center_span(4, 0.01e-3, n=101),
        misc.center_span(4, 10e-3, n=101),
    )
    freq = np.sort(freq)
    s21m1_cplx = s21m1(freq)
    s21_cplx = 1 / s21m1_cplx
    s21_dB = 20 * np.log10(np.abs(s21_cplx))
    s21_rad = np.angle(s21_cplx)
    rfit = FitResonator(freq, s21_dB, s21_rad)
    # rfit.result = None
    rfit.plot()
    # rfit.result.plot_residuals()
    rfit.result
    plt.show()
