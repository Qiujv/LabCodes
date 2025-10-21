# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import labcodes.frog.routine as rt
import labcodes.frog.tele_swep as tele
from labcodes import fileio, misc, plotter, tomo

DIR = '//XLD2-PC2/labRAD_data/crab.dir/220724.dir/661_0814.dir'
# fileio.labrad.browse(DIR)[90:100]

# %%
lf229 = fileio.read_labrad(DIR, 229)
ro_mat = rt.plot_ro_mat(lf229, ax=plt.subplots(figsize=(4,4))[1])
ro_mat

# %%
lf230 = fileio.read_labrad(DIR, 230)
lf230.df

# %%
rho_old, fname, ax_old = rt.plot_qst(DIR, 230)
rho_old, fname, ax_old = rt.plot_qst(DIR, 230, ro_mat=rt.plot_ro_mat(lf229))

# %%
chi, rho_in, rho_out, fname, ax = rt.plot_qpt(
    dir=DIR,
    ro_mat_out=rt.plot_ro_mat(fileio.LabradRead(DIR, 185)),
    out_ids={k: v for k, v in zip('xy10', range(181,185))},
    ro_mat_in=rt.plot_ro_mat(fileio.LabradRead(DIR, 186)),
    in_ids={k: v for k, v in zip('xy10', range(187,191))},
)
