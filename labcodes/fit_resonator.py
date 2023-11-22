import logging

import lmfit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.constants as const

from labcodes import misc, plotter

logger = logging.getLogger(__file__)


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
    n = 2 / (const.hbar * wr**2) * Qc * (Qi / (Qi + Qc))**2 * p_watt
    return n


class FitResonator:
    """Fit the s21 data of linear resonator side loaded to feedline.
    
    Adapted from:
    1. https://lmfit.github.io/lmfit-py/examples/example_complex_resonator_model.html
    2. programs by Xiayu Linpeng.
    """
    def __init__(self, freq: np.ndarray, s21_dB: np.ndarray, s21_rad: np.ndarray, **fit_kws):
        """fit(Qi=1e6) to overwrite initial guess."""
        # Normalize data with y0=0.
        s21_dB = misc.remove_background(s21_dB, freq, fit_mask=100, offset=0)
        s21_rad = misc.remove_background(s21_rad, freq, fit_mask=100, offset=0)
        s21_cplx = 10 ** (s21_dB / 20) * np.exp(1j * s21_rad)
        s21m1_cplx = 1 / s21_cplx
        self.df = pd.DataFrame(
            dict(
                freq=freq,
                s21_dB=s21_dB,
                s21_rad=s21_rad,
                s21_cplx=s21_cplx,
                s21m1_cplx=s21m1_cplx,
            )
        )
        self.params = None
        self.guess_and_update_params()
        self.result: lmfit.minimizer.MinimizerResult = None
        try:
            self.fit(**fit_kws)
        except:
            logger.exception("Error in fitting.")

    def fit(self, **fit_kws) -> lmfit.minimizer.MinimizerResult:
        """fit(Qi=1e6) to overwrite initial guess."""
        freq = self.df.freq.values
        s21m1_cplx = self.df.s21m1_cplx.values

        self.result = model.fit(
            s21m1_cplx,
            x=freq,
            params=self.params,
            weights=np.abs(s21m1_cplx) ** 2,
            method="Powell",
            **fit_kws,
        )
        return self.result

    def guess_and_update_params(self):
        freq = self.df.freq.values
        s21_dB = self.df.s21_dB.values
        s21_cplx = self.df.s21_cplx.values
        s21m1_cplx = self.df.s21m1_cplx.values

        idip = np.argmin(s21_dB)
        fr = freq[idip]

        # Guess Qc using Lorentz fit of S21
        half_maximum = s21_dB[idip] + 3
        fwhm = 0.2 * np.abs(  # Emprical value.
            freq[np.argmin(np.abs(s21_dB[:idip] - half_maximum))]
            - freq[np.argmin(np.abs(s21_dB[idip:] - half_maximum))]
        )
        Qc = np.abs(fr / fwhm)

        # Guess Qi using Lorentz fit of S21^-1
        half_maximum = (np.abs(s21m1_cplx[idip]) + 1) / 2
        fwhm = 0.1 * np.abs(  # Emprical value.
            freq[np.argmin(np.abs(s21m1_cplx[:idip]) - half_maximum)]
            - freq[np.argmin(np.abs(s21m1_cplx[idip:]) - half_maximum)]
        )
        Qi = np.abs(fr / fwhm)
        if np.mean(np.real(s21m1_cplx)) < 1:
            Qi = -Qi

        alpha = fr * (abs(s21_cplx[-1]) - abs(s21_cplx[0])) / (freq[-1] - freq[0])
        phiv = np.angle(s21_cplx[-1] / s21_cplx[0]) / (freq[-1] - freq[0])

        params = lmfit.Parameters()
        params.set(Qi=Qi, Qc=Qc, f0=fr, phi=0, alpha=alpha, phiv=phiv, phi0=0, amp=1)
        self.params = params
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
            return self.params[key].value
        
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
        plot_init: bool = False,
        plot_annotate: bool = True,
        plot_orig: bool = True,
    ):
        if ax is None:
            _, ax = plt.subplots(figsize=(3, 3), layout="compressed")
            ax.set_xlabel("Re[$S_{21}^{-1}$]")
            ax.set_ylabel("Im[$S_{21}^{-1}$]")
        if freq is None:
            freq = self.df.freq.values

        f0 = self["f0"]
        Qi = self["Qi"]
        Qc = self["Qc"]
        dfi = f0 / Qi / 2
        dfc = f0 / Qc / 2

        def plot(ax: plt.Axes, x: np.ndarray, y: np.ndarray, sty="-", **kw):
            dx = np.abs(x - f0)
            mask1 = dx <= dfi
            mask3 = dx >= dfi * 10
            mask2 = np.logical_not(mask1 | mask3)
            ax.plot(y.real[mask1], y.imag[mask1], sty, color="C0", **kw)
            ax.plot(y.real[mask2], y.imag[mask2], sty, color="C1", **kw)
            ax.plot(y.real[mask3], y.imag[mask3], sty, color="C2", **kw)

        ax.set_aspect("equal")

        plot(ax, self.df.freq.values, self.df.s21m1_cplx.values, ".", alpha=0.5)
        if self.result is not None:
            plot(ax, freq, self.result.eval(x=freq), "-")
        if plot_init:
            init_param = {k: p.init_value for k, p in self.result.params.items()}
            plot(ax, freq, model.eval(x=freq, **init_param), "--")

        if plot_annotate:
            ax.annotate(
                f"$f_0$={misc.estr(f0)}\n$Q_i$={misc.estr(Qi)}\n$Q_c$={misc.estr(Qc)}",
                (0.5, 0.5),
                xycoords="axes fraction",
                ha="center",
                va="center",
            )
        if plot_orig:
            ax.axhline(y=0, color="gray", alpha=0.5, zorder=1)
            ax.axvline(x=1, color="gray", alpha=0.5, zorder=1)
        return ax

    def plot_s21(
        self,
        axs: tuple[plt.Axes, plt.Axes] = None,
        freq: np.ndarray = None,
        plot_init: bool = False,
    ):
        if axs is None:
            _, (ax, ax2) = plt.subplots(figsize=(6, 2), ncols=2)
            ax.set_xlabel("freq")
            ax.set_ylabel("s21_dB")
            ax2.set_xlabel("freq")
            ax2.set_ylabel("s21_rad")
        else:
            ax, ax2 = axs
        ax2.set_ylim(-3.2, 3.2)
        ax2.yaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
        ax2.yaxis.set_major_formatter(plotter.misc.multiple_formatter(2, np.pi))

        if freq is None:
            freq = self.df.freq.values

        f0 = self["f0"]
        Qi = self["Qi"]
        Qc = self["Qc"]
        dfi = f0 / Qi / 2
        dfc = f0 / Qc / 2

        def plot(ax: plt.Axes, x: np.ndarray, y: np.ndarray, sty="-"):
            dx = np.abs(x - f0)
            mask1 = dx <= dfi
            mask3 = dx >= dfi * 10
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
        plot(ax2, self.df.freq.values, self.df.s21_rad.values, ".")

        if self.result is not None:
            s21_cplx = 1 / self.result.eval(x=freq)
            ax.plot(freq, 20 * np.log10(np.abs(s21_cplx)), "k-")
            ax2.plot(freq, np.angle(s21_cplx), "k-")

        if plot_init:
            init_param = {k: p.init_value for k, p in self.result.params.items()}
            s21_cplx = 1 / model.eval(x=freq, **init_param)
            ax.plot(freq, 20 * np.log10(np.abs(s21_cplx)), "--", color="gray")
            ax2.plot(freq, np.angle(s21_cplx), "--", color="gray")

        return ax, ax2

    def plot(self, title: str = None, axs: tuple[plt.Axes, plt.Axes, plt.Axes] = None):
        if axs is None:
            _, (ax, ax2, ax3) = plt.subplots(
                figsize=(8, 3), ncols=3, layout="constrained"
            )
            ax.set_xlabel("freq")
            ax.set_title("s21_dB")
            ax2.set_xlabel("freq")
            ax2.set_title("s21_rad")
            ax3.set_xlabel("Re[$S_{21}^{-1}$]")
            ax3.set_ylabel("Im[$S_{21}^{-1}$]")
        self.plot_cplx(ax=ax3)
        self.plot_s21(axs=(ax, ax2))
        ax3.set_title(title)
        return axs


if __name__ == "__main__":
    from labcodes import TEST_DATA_PATH

    df = pd.read_feather(TEST_DATA_PATH / "fit_resonator.feather")
    rfit = FitResonator(df.freq_GHz, df.s21_mag_dB, df.s21_phase_rad)
    rfit.plot()
    rfit.result
    plt.show()
