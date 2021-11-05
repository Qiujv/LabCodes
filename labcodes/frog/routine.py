"""Script provides functions dealing with routine experiment datas."""


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from labcodes import misc, fitter, models, plotter


def prep_data_one(logf, atten=0):
    """Normalize S21 data for models.ResonatorModel_inverse."""
    df = logf.df.copy()
    angle = misc.remove_e_delay(df['s21_rad'].values, df['freq_GHz'].values)
    df['s21'] = 10 ** (df['s21_dB'] / 20) * np.exp(1j*angle)
    df['s21'] = models.ResonatorModel_inverse.normalize(df['s21'])
    df['1_s21'] = 1/df['s21']
    df['id'] = int(logf.name[:5])
    # df['bw_Hz'] = float(logf.config['Parameter 2']['data'][6:-7])
    df['bw_kHz'] = float(logf.config['Parameter 2']['data'][6:-8])
    df['power_dBm'] = float(logf.config['Parameter 3']['data'][6:-8]) - atten
    df['frr_GHz'] = df['freq_GHz'].mean()
    return df

def fit_resonator(logf, atten=0, **kwargs):
    df = prep_data_one(logf, atten)
    cfit = fitter.CurveFit(
        xdata=df['freq_GHz'].values,
        ydata=df['1_s21'].values,
        model=models.ResonatorModel_inverse(),
        hold=True,
    )
    cfit.fit(**kwargs)
    ax = cfit.plot_complex(plot_init=False, fit_report=True)
    return cfit, ax
    
def fit_coherence(logf, ax, model=None, xy=(0.6,0.9)):
    if 'T1' in logf.name:
        mod = models.ExponentialModel()
        symbol = 'T_1'
    elif 'Ramsey' in logf.name:
        mod = models.ExpSineModel()
        symbol = 'T_2^*'
    elif 'Echo' in logf.name:
        mod = models.ExpSineModel()
        symbol = 'T_{2e}'
    else:
        mod = models.ExponentialModel()
        symbol = '\\tau'
    if model:
        mod = model
    for indep in logf.indeps:
        if indep.startswith('delay'):
            xname = indep

    cfit = fitter.CurveFit(
        xdata=logf.df[xname].values,
        ydata=logf.df['s1_prob'].values,
        model=mod,
    )

    fig = ax.get_figure()
    fig.set_size_inches(5,3)

    ax.plot(*cfit.fdata(100), 'r-', lw=1)
    ax.annotate(f'${symbol}\\approx {cfit["tau"]:,.2f}\\pm{cfit["tau_err"]:,.4f} {xname[-2:]}$', 
        xy, xycoords='axes fraction')
    return cfit, ax


def fit_spec(spec_map, logf, ax=None):
    cfit = fitter.CurveFit(
        xdata=np.array(list(spec_map.keys())),
        ydata=np.array(list(spec_map.values())),
        model=models.TransmonModel()
    )
    ax = cfit.model.plot(cfit, ax=ax)
    ax.set(
        xlabel=logf.indeps[0],
        ylabel='Frequency (GHz)',
        title=logf._get_plot_title(),
    )
    return cfit, ax

def plot_visibility(logf, axs=None, drop=True, **kwargs):
    """Plot visibility, for iq_scatter experiments only."""
    if axs is None:
        fig = plt.figure(figsize=(6,6), tight_layout=True)
        ax, ax2, ax3 = fig.add_subplot(221), fig.add_subplot(222), fig.add_subplot(212)
        ax4 = ax3.twinx()
    else:
        ax, ax2, ax3, ax4 = axs
        fig = ax.get_figure()

    # Make up single shot datas.
    df = logf.df
    df['s0'] = df['i0'] + 1j*df['q0']
    df['s1'] = df['i1'] + 1j*df['q1']
    if drop is True:
        df.drop(columns=['runs', 'i0', 'q0', 'i1', 'q1'], inplace=True)

    fig.suptitle(logf._get_plot_title())
    plotter.plot_iq(df['s0'], ax=ax, label='|0>')  # The best plot maybe PDF contour plot with colored line.
    plotter.plot_iq(df['s1'], ax=ax, label='|1>')
    ax.legend()

    df[['s0_rot', 's1_rot']] = misc.auto_rotate(df[['s0', 's1']].values)  # Must pass np.array.
    plotter.plot_iq(df['s0_rot'], ax=ax2, label='|0>')
    plotter.plot_iq(df['s1_rot'], ax=ax2, label='|1>')
    ax2.legend()

    plotter.plot_visibility(np.real(df['s0_rot']), np.real(df['s1_rot']), ax3, ax4)

    return ax, ax2, ax3, ax4


class GmonModel(models.MyCompositeModel):
    """Model fitting Gmon induced tunable coupling.
    WARNING: The fit is sensitive to initial value, which must be provided by user."""

    def __init__(self, independent_vars=['x'], prefix='', nan_policy='raise',
                 **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})
       
        def coupling(x, r=0.9, freq=1, shift=0):
            delta = junc_phase(2*np.pi*freq*(x-shift),r)
            M = 1 / (r + 1/np.cos(delta))
            return M

        mod = models.AmpFeature() * models.MyModel(coupling)
        mod.set_param_hint(name='r', max=1, min=0)  # r = L_linear / L_j0, see my notes.
        mod.set_param_hint(name='amp', min=0)
        mod.set_param_hint(name='zero1', expr='(pi/2+r)/(2*pi*freq) + shift')
        mod.set_param_hint(name='zero2', expr='(pi*3/2-r)/(2*pi*freq) + shift')
        mod.set_param_hint(name='period', expr='1/freq')
        mod.set_param_hint(name='max_y_shift', expr='amp/(r-1)')

        super().__init__(mod, models.OffsetFeature(), models.operator.add, **kwargs)

    __init__.__doc__ = 'Gmon model' + models.COMMON_INIT_DOC

    def plot(self, cfit, ax=None, fdata=True):
        """Plot fit with results parameters."""
        if ax is None:
            fig, ax = plt.subplots(tight_layout=True)
        else:
            fig = ax.get_figure()

        if fdata is True:
            ax.plot(cfit.xdata, cfit.ydata, 'o')
            ax.plot(*cfit.fdata(50))

        gs = dict(ls='--', color='k', alpha=0.5)  # Guide line style
        ax.axhline(y=cfit['offset'], **gs)
        ax.annotate(f"y0={cfit['offset']:.3f}", 
            (ax.get_xlim()[0], cfit['offset']), ha='left')

        dip_x = (cfit['zero1']+cfit['zero2'])/2
        dip_y = cfit['offset'] + cfit['max_y_shift']
        ax.axhline(y=dip_y, **gs)
        ax.axvline(x=dip_x, **gs)
        ax.annotate(f"dip: {dip_x:.3f}\nmax y shift: {cfit['max_y_shift']:.4f}", 
            (dip_x, dip_y), ha='center')

        ax.axvline(x=cfit['shift'], **gs)
        ax.annotate(f"x={cfit['shift']:.3f}", 
            (cfit['shift'], ax.get_ylim()[1]), va='top', ha='center')

        ax.axvline(x=cfit['zero1'], **gs)
        ax.annotate(f"x={cfit['zero1']:.3f}", 
            (cfit['zero1'], cfit['offset']), va='bottom', ha='center')

        ax.axvline(x=cfit['zero2'], **gs)
        ax.annotate(f"x={cfit['shift']:.3f}", 
            (cfit['zero2'], cfit['offset']), va='bottom', ha='center')

        ax.set(
            xlabel='x',
            ylabel='y'
        )
        return ax

def junc_phase(delta_ext, r):
    # fsolve does not works with np.array.
    if isinstance(delta_ext, np.ndarray):
        # Solve the values one by one instead of a high-dimensional system (it is decoupled).
        delta = [fsolve(lambda d: de - _delta_ext(d, r), 0)[0] 
                for de in delta_ext.ravel()]
        delta = np.array(delta).reshape(delta_ext.shape)
    else:
        delta = fsolve(lambda d: delta_ext - _delta_ext(d, r), 0)[0]
    return delta

def _delta_ext(delta, r):
    return delta + np.sin(delta) * r