"""Script provides functions dealing with routine experiment datas."""


import numpy as np
import matplotlib.pyplot as plt
from labcodes import misc, fitter, models, plotter, fileio

from labcodes.models import GmonModel  # TODO: remove this.


def plot_yourself(logf):
    """Choosing plot function according to logf.name. Returns depends on input.
    User should know exactly what they are doing."""
    pass

def plot2d_multi(dir, ids, sid=None, title=None, x_name=0, y_name=1, z_name=0, ax=None, **kwargs):
    """Plot 2d with data from multiple logfiles.
    Args:
        sid, title: str, information shown in title and file name.
        **kwargs: passed to lf.plot2d()

    Returns:
        the axes, list of logfiles and str of file name.
    """
    lfs = [fileio.LabradRead(dir, id) for id in ids]
    lf = lfs[0]

    if sid is None: sid = f'{ids[0]}-{ids[-1]}'
    if title is None: title = lf.name.title
    if isinstance(x_name, int):
        x_name = lf.indeps[x_name]
    if isinstance(y_name, int):
        y_name = lf.indeps[y_name]
    if isinstance(z_name, int):
        z_name = lf.deps[z_name]

    cmin = np.min([lf.df[z_name].min() for lf in lfs])
    cmax = np.max([lf.df[z_name].max() for lf in lfs])
    plot_kw = dict(x_name=x_name, y_name=y_name, z_name=z_name, cmin=cmin, cmax=cmax)
    plot_kw.update(kwargs)
    ax = lf.plot2d(ax=ax, **plot_kw)
    for lf in lfs[1:]:
        lf.plot2d(ax=ax, colorbar=False, **plot_kw)
    ax.set_title(lf.name.as_plot_title(id=sid, title=title))
    fname = lf.name.as_file_name(id=sid, title=title)
    return ax, lfs, fname

def prep_data_one(logf, atten=0):
    """Normalize S21 data for models.ResonatorModel_inverse."""
    df = logf.df.rename(columns={'s21_log_mag_dB': 's21_dB', 's21_phase_rad': 's21_rad'})
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

def fit_resonator(logf, atten=0, fdata=500, **kwargs):
    df = prep_data_one(logf, atten)
    cfit = fitter.CurveFit(
        xdata=df['freq_GHz'].values,
        ydata=df['1_s21'].values,
        model=models.ResonatorModel_inverse(),
        hold=True,
    )
    cfit.fit(**kwargs)
    ax = cfit.model.plot(cfit, fdata=fdata)
    ax.set_title(logf.name.as_plot_title())
    return cfit, ax
    
def fit_coherence(logf, ax, model=None, xy=(0.6,0.9), fdata=500, kind=None, **kwargs):
    if kind is None: kind = str(logf.name)
    if 'T1' in kind:
        mod = models.ExponentialModel()
        symbol = 'T_1'
    elif 'Ramsey' in kind:
        mod = models.ExpSineModel()
        symbol = 'T_2^*'
    elif 'Echo' in kind:
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
        hold=True,
    )
    cfit.fit(**kwargs)

    fig = ax.get_figure()
    fig.set_size_inches(5,3)

    ax.plot(*cfit.fdata(fdata), 'r-', lw=1)
    ax.annotate(f'${symbol}\\approx {cfit["tau"]:,.2f}\\pm{cfit["tau_err"]:,.4f} {xname[-2:]}$', 
        xy, xycoords='axes fraction')
    return cfit, ax


def fit_spec(spec_map, ax=None, **kwargs):
    cfit = fitter.CurveFit(
        xdata=np.array(list(spec_map.keys())),
        ydata=np.array(list(spec_map.values())),
        model=models.TransmonModel()
    )
    ax = cfit.model.plot(cfit, ax=ax, **kwargs)
    return cfit, ax

def plot_visibility(logf, axs=None, drop=True, **kwargs):
    """Plot visibility, for iq_scatter experiments only."""
    if axs is None:
        fig = plt.figure(figsize=(5,5), tight_layout=True)
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

    fig.suptitle(logf.name.as_plot_title())
    plotter.plot_iq(df['s0'], ax=ax, label='|0>')  # The best plot maybe PDF contour plot with colored line.
    plotter.plot_iq(df['s1'], ax=ax, label='|1>')
    # ax.legend()

    df[['s0_rot', 's1_rot']] = misc.auto_rotate(df[['s0', 's1']].values)  # Must pass np.array.
    if df['s0_rot'].mean().real > df['s1_rot'].mean().real:
        # Flip if 0 state cloud is on the right.
        df[['s0_rot', 's1_rot']] *= -1
    plotter.plot_iq(df['s0_rot'], ax=ax2, label='|0>')
    plotter.plot_iq(df['s1_rot'], ax=ax2, label='|1>')
    # ax2.legend()

    plotter.plot_visibility(np.real(df['s0_rot']), np.real(df['s1_rot']), ax3, ax4)

    return ax, ax2, ax3, ax4

def plot_iq_vs_freq(logf, axs=None):
    if axs is None:
        fig, (ax, ax2) = plt.subplots(tight_layout=True, figsize=(5,5), nrows=2, sharex=True)
        ax3 = ax2.twinx()
    else:
        ax, ax2, ax3 = axs
        fig = ax.get_figure()
    df = logf.df
    ax.plot(df['ro_freq_MHz'], df['iq_amp_(0)'], label='|0>')
    ax.plot(df['ro_freq_MHz'], df['iq_amp_(1)'], label='|1>')
    ax.grid(True)
    ax.legend()
    ax.set(
        ylabel='IQ amp',
    )

    ax2.plot(df['ro_freq_MHz'], df['iq_difference_(0-1)'])
    ax2.grid(True)
    ax2.set(
        ylabel='IQ diff',
        xlabel='RO freq (MHz)'
    )
    ax3.plot(df['ro_freq_MHz'], df['iq_snr'], color='C1')
    # ax3.grid(True)
    ax3.set_ylabel('SNR', color='C1')
    fig.suptitle(logf.name.as_plot_title())
    return ax, ax2, ax3

def plot2d_ruler(logf, slope=-0.01, offset=0.0, **kwargs):
    """Plot 2d with a guide line.
    For xtalk data.
    
    Args:
        slope, offset: float, property of the guide line.
        kwargs: passed to logf.plot2d.
    
    """
    ax = logf.plot2d(**kwargs)

    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    c = np.mean(xlims), np.mean(ylims)
    x = np.linspace(*xlims)
    y = slope*(x-c[0]) + c[1] + offset*(ylims[1]-ylims[0])/2
    mask = (y>ylims[0]) & (y<ylims[1])
    ax.plot(x[mask], y[mask], lw=3, color='k')
    ax.annotate(f'{slope*100:.2f}%', c, size='xx-large', ha='left', va='bottom', 
        bbox=dict(facecolor='w', alpha=0.7, edgecolor='none'))
    return ax

def plot_cramsey(cfit0, cfit1, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(cfit0.xdata, cfit0.ydata, 'o', color='C0')
    ax.plot(cfit1.xdata, cfit1.ydata, 'x', color='C1')
    ax.plot(*cfit0.fdata(100), color='C0', label='Ctrl 0')
    ax.plot(*cfit1.fdata(100), color='C1', label='Ctrl 1')
    def mark_maxi(ax, cfit, **kwargs):
        shift = (np.pi/2 - cfit['phase']) / (2*np.pi*cfit['freq'])
        for x in misc.multiples(0.5/cfit['freq'], shift, cfit.xdata.min(), cfit.xdata.max()):
            ax.axvline(x, **kwargs)
    mark_maxi(ax, cfit0, ls='--', color='C0', alpha=0.5)
    mark_maxi(ax, cfit1, ls='--', color='C1', alpha=0.5)
    ax.legend()
    ax.grid(True)
    return ax

def plot_ro_mat(logf, states=('gg', 'ge', 'eg', 'ee'), ax=None, check_label=False, txt_color=('white', 'black')):
    if ax is None:
        _, ax = plt.subplots()
    n_sts = np.size(states)
    mat_vals = logf.df.mean()  # pandas.Series, with index as channel names.
    if np.size(mat_vals) != n_sts**2:
        mat_vals = mat_vals.iloc[1:]  # 1st column or df is #run.
    mat = np.reshape(mat_vals.values, (n_sts, n_sts))
    im = ax.matshow(mat)
    
    if check_label is True:
        txt_mat = np.reshape(mat_vals.index.values, (n_sts, n_sts))
        fmt = '{}'.format
    else:
        txt_mat = mat*100  # Percentage.
        fmt = '{:.1f}%'.format
    threshold = (im.norm.vmax + im.norm.vmin) / 2
    for i in range(n_sts):
        for j in range(n_sts):
            color = txt_color[int(mat[i,j] > threshold)]
            ax.annotate(fmt(txt_mat[i,j]), (i, j), ha='center', va='center', color=color)
    ax.set(
        title=logf.name.as_plot_title(),
        xlabel='Prepare',
        xticks=np.arange(n_sts),
        xticklabels=states,
        yticks=np.arange(n_sts),
        ylabel='Measure',
        yticklabels=states,
    )
    return ax, mat