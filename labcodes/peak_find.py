import logging
import warnings

import lmfit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal
from labcodes.fitter import getitem_from_fit_result

logger = logging.getLogger(__name__)


def lorentzian(x, prominence, center, hwhm, offset):
    # From https://lmfit.github.io/lmfit-py/builtin_models.html#lmfit.models.LorentzianModel
    
    # tiny had been numpy.finfo(numpy.float64).eps ~=2.2e16.
    # here, we explicitly set it to 1.e-15 == numpy.finfo(numpy.float64).resolution
    tiny = 1.0e-15

    sigma = max(hwhm, tiny)
    return prominence / (1 + ((x - center) / sigma) ** 2) + offset


class PeakFinder:
    def __init__(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)

        if not np.all(np.diff(x) > 0):
            idx = np.argsort(x)
            x = x[idx]
            y = y[idx]

        self.x = x
        self.y = y
        self.model = lmfit.Model(lorentzian)
        self.init_params = None
        self.init_params = self.guess()
        self.result: lmfit.model.ModelResult = None

        try:
            self.fit()
        except Exception:
            logger.exception("Error in fitting.")

    def fit(self, **fit_kws):
        self.result = self.model.fit(
            self.y,
            x=self.x,
            params=self.init_params,
            **fit_kws,
        )
        return self.result
    
    def guess(self, peak: dict[str, float]=None):
        if peak is None:
            peak = self.peaks().sort_values("prominence", ascending=False).iloc[0]

        return self.model.make_params(
            prominence=peak["prominence"],
            center=peak["x"],
            hwhm=peak["hm_fw"] / 2,
            offset=peak["base_y0"],
        )

    def peaks(self, **kwargs):
        """Find peaks in the data. kwargs passed to `scipy.signal.find_peaks`."""
        x = self.x
        y = self.y

        peaks, _ = scipy.signal.find_peaks(y, **kwargs)
        if len(peaks) == 0:
            warnings.warn("No peaks found")
        width, width_heights, hm_i0, hm_i1 = scipy.signal.peak_widths(y, peaks)
        prominence, base_i0, base_i1 = scipy.signal.peak_prominences(y, peaks)
        hm_x0 = x[hm_i0.astype(int)]
        hm_x1 = x[hm_i1.astype(int)]
        return pd.DataFrame(
            {
                "x": x[peaks],
                "y": y[peaks],
                "hm_fw": hm_x1 - hm_x0,
                "hm_y": width_heights,
                "hm_x0": hm_x0,
                "hm_x1": hm_x1,
                "hm_i0": hm_i0,
                "hm_i1": hm_i1,
                "hm_wi": width,
                "prominence": prominence,
                "base_y0": y[base_i0],
                "base_y1": y[base_i1],
                "base_i0": base_i0,
                "base_i1": base_i1,
            }
        )

    def show_peaks(self, peaks: pd.DataFrame = None, ax: plt.Axes = None):
        if peaks is None:
            peaks = self.peaks()

        if ax is None:
            _, ax = plt.subplots()
            ax.set_xlabel("x")
            ax.set_ylabel("y")

        ax.plot(self.x, self.y)
        ax.plot(peaks["x"], peaks["y"], "x")
        for i, row in peaks.iterrows():
            ax.hlines(row["hm_y"], row["hm_x0"], row["hm_x1"], color="red")
            ax.vlines(row["x"], row["y"] - row["prominence"], row["y"], color="red")

        return ax
    
    def __getitem__(self, k: str):
        return getitem_from_fit_result(self.result, k)
