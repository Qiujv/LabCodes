# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
from attrs import define, field

from labcodes import fileio, fitter, models, misc, plotter


@define(slots=False)
class IQScatter:
    name: fileio.LogName
    lf: fileio.LogFile
    df: pd.DataFrame
    angle: float
    bins: int = 50
    thres: float = field(init=False)
    p00: float = field(init=False)
    p11: float = field(init=False)
    df_hist: pd.DataFrame = field(init=False)
    def __attrs_post_init__(self):
        i0 = self.df['c0_rot'].values.real
        i1 = self.df['c1_rot'].values.real

        # Calc histograms.
        bins = np.histogram_bin_edges(np.array([i0, i1]).real, bins=self.bins)
        pdf0, _ = np.histogram(i0, bins, density=True)
        pdf1, _ = np.histogram(i1, bins, density=True)
        x = (bins[:-1] + bins[1:]) / 2
        dx = np.diff(bins)
        cdf0 = np.cumsum(pdf0 * dx)
        cdf1 = np.cumsum(pdf1 * dx)
        visi = abs(cdf1 - cdf0)
        self.df_hist = pd.DataFrame(dict(x=x, pdf0=pdf0, cdf0=cdf0, pdf1=pdf1, cdf1=cdf1, visi=visi))
        self.bins = bins

        argmax = np.argmax(visi)
        self.thres = x[argmax]
        self.p00 = cdf0[argmax]
        self.p11 = 1 - cdf1[argmax]

        # Fit histogram with Gaussian model & extract seperation error.
        mod = models.GaussianModel()
        mod.set_param_hint('amp', expr='1/(width*sqrt(2*pi))')
        mod.set_param_hint('offset', vary=False)
        cfit0 = fitter.CurveFit(xdata=x, ydata=pdf0, model=mod, hold=True)
        cfit0.fit(offset=0)
        cfit1 = fitter.CurveFit(xdata=x, ydata=pdf1, model=mod, hold=True)
        cfit1.fit(offset=0)
        xlin = np.linspace(cfit0['center'], cfit1['center'], 100000)
        thres = xlin[np.argmin(np.abs(cfit0.fdata(xlin)[1] - cfit1.fdata(xlin)[1]))]
        self.p_resi = 1 - scipy.stats.norm(cfit0['center'], cfit0['width']).cdf(thres)
        self.cfit0 = cfit0
        self.cfit1 = cfit1
        self.p00 = (i0 <= thres).mean()
        self.p11 = (i1 >  thres).mean()
        self.thres = thres
    
    @classmethod
    def from_logfile(cls, lf, bins=50):
        df = lf.df.copy()
        df['c0'] = df['i0'] + 1j*df['q0']
        df['c1'] = df['i1'] + 1j*df['q1']
        df.drop(columns=['runs', 'i0', 'q0', 'i1', 'q1'], inplace=True)

        df[['c0_rot', 'c1_rot']], rad = misc.auto_rotate(df[['c0', 'c1']].values,
                                                         return_rad=True)
        if df['c0_rot'].mean().real > df['c1_rot'].mean().real:
            # Flip if 0 state cloud is on the right.
            df[['c0_rot', 'c1_rot']] *= -1
            rad = rad + np.pi
        return cls(lf=lf, df=df, name=lf.name.copy(), angle=rad, bins=bins)

    @classmethod
    def from_data(cls, c0, c1, bins=50):
        (c0_rot, c1_rot), rad = misc.auto_rotate(np.array([c0, c1]), return_rad=True)
        if c0_rot.real.mean() > c1_rot.real.mean():
            c0_rot = -1 * c0_rot
            c1_rot = -1 * c1_rot
            rad = rad + np.pi
        df = pd.DataFrame(dict(c0=c0, c1=c1, c0_rot=c0_rot, c1_rot=c1_rot))
        name = fileio.LogName('.', id='', title='')
        return cls(lf=None, df=df, name=name, angle=rad, bins=bins)

    def plot(self):
        lr = self
        df = self.df

        fig = plt.figure(figsize=(5,5), tight_layout=True)
        ax, ax2, ax3 = fig.add_subplot(221), fig.add_subplot(222), fig.add_subplot(212)
        ax.sharex(ax2)
        ax.sharey(ax2)
        ax4 = ax3.twinx()

        fig.suptitle(lr.name.as_plot_title())
        ax.set_title('|0>', color='C0')
        ax2.set_title('|1>', color='C1')
        ax.axvline(x=self.thres, ls='--', color='k')
        ax2.axvline(x=self.thres, ls='--', color='k')
        plotter.plot_iq(df['c0_rot'], ax=ax , label='|0>', color='C0')
        plotter.plot_iq(df['c1_rot'], ax=ax2, label='|1>', color='C1')
        ax.annotate(f'{self.p00:.1%}', (0.1, 0.9), xycoords='axes fraction')
        ax2.annotate(f'{self.p11:.1%}', (0.6, 0.9), xycoords='axes fraction')
        ax.set(xlabel='', ylabel='')
        ax2.set(xlabel='', ylabel='')

        i0 = self.df['c0_rot'].values.real
        i1 = self.df['c1_rot'].values.real
        # ax3.hist(i0, bins=self.bins, density=True, cumulative=True, histtype='step')
        # ax3.hist(i1, bins=self.bins, density=True, cumulative=True, histtype='step')
        ax3.step(self.bins[:-1], self.df_hist['cdf0'], where='post')
        ax3.step(self.bins[:-1], self.df_hist['cdf1'], where='post')
        ax3.step(self.bins[:-1], self.df_hist['visi'], where='post')
        ax3.axvline(x=self.thres, ls='--', color='k')
        visi = self.p00 + self.p11 - 1
        ax3.annotate(f'best visibility = {visi:.1%}', (self.thres, visi), 
                     ha='center', va='top')
        ax3.annotate(f'seperation error = {self.p_resi:.2%}', (self.thres, self.p_resi),
                     ha='center', va='bottom')
        ax3.set(
            xlabel='Quadrature',
            ylabel='Probability',
            ylim=(0,1),
            yticks=[0,0.5,1],
        )

        ax4.hist(i0, bins=self.bins, density=True, alpha=0.6)
        ax4.hist(i1, bins=self.bins, density=True, alpha=0.6)
        ax4.plot(*self.cfit0.fdata(500), color='k')
        ax4.plot(*self.cfit1.fdata(500), color='k')
        ax4.set(
            ylabel='Probability Density',
            yticks=[],
        )
        return ax, ax2, ax3, ax4


if __name__ == '__main__':
    rng = np.random.default_rng()
    std = 0.25
    noise = std * rng.standard_normal(5000)
    noise2 = std * rng.standard_normal(5000)
    c0 = (0+noise) + 1j*(0+noise2)
    c1 = (1+noise) + 1j*(1+noise2)
    c0 = np.hstack([c1[::100], c0])[:c0.size]
    c1 = np.hstack([c0[::50], c1])[:c0.size]
    
    lr = IQScatter.from_data(c0, c1)
    lr.plot()

    # from labcodes import fileio
    # DIR = '//XLD2-PC2/labRAD_data/crab.dir/221203.dir/1203_bup.dir'
    # DIR = 'C://Users/qiujv/downloads'

    # lf = fileio.LabradRead(DIR, 96)
    # lr = IQScatter.from_logfile(lf)
    # lr.plot()
