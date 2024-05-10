# %%
from labcodes import peak_find
import numpy as np
import pandas as pd

pf = peak_find.PeakFinder(
    x=np.linspace(0, 5, 10),
    y=np.array([0.0, 0.05, 0.16, 0.3, 0.6, 0.8, 0.6, 0.3, 0.16, 0.0]),
)
pf.show_peaks()
pf.result.plot(show_init=True)
pf.result

# %%
from scipy.datasets import electrocardiogram

y = electrocardiogram()[17000:18000]
x = np.arange(len(y))
pf = peak_find.PeakFinder(x, y)
peaks = pf.peaks(distance=200).query('prominence > 0.1')
pf.show_peaks(peaks)
_ = pf.result.plot(show_init=True)

pf['center'], pf['hwhm'], pf['prominence_err']

# %%
from labcodes import fit_resonator
df = pd.read_feather('./data/resonator_s21.feather')
rfit = fit_resonator.FitResonator(df=df)

pf = peak_find.PeakFinder(rfit.df.freq, -np.abs(rfit.s21_cplx))
pf.show_peaks()
pf.result.plot(show_init=True)
pf.result
