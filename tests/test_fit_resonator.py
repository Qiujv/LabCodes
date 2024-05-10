# %%
import pandas as pd
from labcodes import fit_resonator

df = pd.read_feather('./data/resonator_s21.feather')
rfit = fit_resonator.FitResonator(df=df)
rfit.plot()
rfit.result
