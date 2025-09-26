import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from attrs import define, field

from labcodes import fileio, fitter, models


def rb_decay(x, amp=0.5, fid=0.99, residue=0.0):
    return amp * fid**x + residue

@define
class XEB:
    folder: str
    id: int
    lf: field()
    df: field()
    cfit: field()

    @classmethod
    def from_logfile(cls, lf):
        df = lf.df.groupby('m').mean().reset_index().query('m > 1')
        cfit = fitter.CurveFit(
            xdata=df['m'].values,
            ydata=df['fidelity_per_round'].values,
            model=models.Model(rb_decay),
        )
        df['fitted_y'] = cfit.fdata(x=df['m'])[1]

        return cls(folder=lf.path.parent, id=lf.name.id, lf=lf, df=df, cfit=cfit)

    def plot(self):
        cfit = self.cfit

        fig, ax = plt.subplots()
        ax.plot(cfit.xdata, cfit.ydata, '.')
        ax.plot(*cfit.fdata(500), color='gray', label=f'{1-cfit["fid"]:.1%} error per cycle')
        ax.annotate('${:.3f}\\times{:.3f}^m {:+.3f}$'.format(cfit['amp'], cfit['fid'], cfit['residue']),
                    (0.55,0.8), xycoords='axes fraction')
        ax.tick_params(direction='in')
        ax.legend()
        ax.set(
            title=self.lf.name.as_plot_title(),
            xlabel='Number of cycles',
            ylabel='XEB fidelity',
        )
        return ax