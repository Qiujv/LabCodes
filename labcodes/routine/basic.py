import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from attrs import define, field

from labcodes import fileio, fitter, models, misc, plotter


@define(slots=False)
class IQScatter:
    name: field()
    lf: field()
    df: field()
    
    @classmethod
    def from_logfile(cls, lf):
        df = lf.df.copy()
        df['s0'] = df['i0'] + 1j*df['q0']
        df['s1'] = df['i1'] + 1j*df['q1']
        df.drop(columns=['runs', 'i0', 'q0', 'i1', 'q1'], inplace=True)

        df[['s0_rot', 's1_rot']] = misc.auto_rotate(df[['s0', 's1']].values)  # Must pass np.array.
        if df['s0_rot'].mean().real > df['s1_rot'].mean().real:
            # Flip if 0 state cloud is on the right.
            df[['s0_rot', 's1_rot']] *= -1

        return cls(lf=lf, df=df, name=lf.name.copy())

    def plot(self):
        lr = self
        df = self.df

        fig = plt.figure(figsize=(5,5), tight_layout=True)
        ax, ax2, ax3 = fig.add_subplot(221), fig.add_subplot(222), fig.add_subplot(212)
        ax4 = ax3.twinx()

        fig.suptitle(lr.name.as_plot_title())
        plotter.plot_iq(df['s0'], ax=ax, label='|0>')  # The best plot maybe PDF contour plot with colored line.
        plotter.plot_iq(df['s1'], ax=ax, label='|1>')
        # ax.legend()

        plotter.plot_iq(df['s0_rot'], ax=ax2, label='|0>')
        plotter.plot_iq(df['s1_rot'], ax=ax2, label='|1>')
        # ax2.legend()

        plotter.plot_visibility(np.real(df['s0_rot']), np.real(df['s1_rot']), ax3, ax4)
        return ax, ax2, ax3, ax4

# %%
from labcodes import fileio
import labcodes.routine.basic as rt

DIR = '//XLD2-PC2/labRAD_data/crab.dir/221203.dir/1203_bup.dir'
lf = fileio.LabradRead(DIR, 226)
lr = rt.IQScatter.from_logfile(lf)
lr.plot()
