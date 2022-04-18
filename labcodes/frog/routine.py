"""Script provides functions dealing with routine experiment datas."""


import numpy as np
import matplotlib.pyplot as plt
from labcodes import misc, fitter, models, plotter, fileio
import labcodes.frog.pyle_tomo as tomo


def plot2d_multi(dir, ids, sid=None, title=None, x_name=0, y_name=1, z_name=0, ax=None, **kwargs):
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

    # Plot with same colorbar.
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

def plot1d_multi(dir, ids, lbs=None, sid=None, title=None, ax=None, **kwargs):
    lfs = [fileio.LabradRead(dir, id) for id in ids]

    if sid is None: sid = f'{ids[0]}-{ids[-1]}'
    if lbs is None: lbs = ids

    for lf, lb in zip(lfs, lbs):
        ax = lf.plot1d(label=lb, ax=ax, **kwargs)
    if title is None: title = lf.name.title
    ax.legend()
    ax.set_title(lf.name.as_plot_title(id=sid, title=title))
    fname = lf.name.as_file_name(id=sid, title=title)
    return ax, lfs, fname

def fit_resonator(logf, axs=None, i_start=0, i_end=-1, annotate='', init=False, **kwargs):
    if axs is None:
        fig, (ax, ax2) = plt.subplots(ncols=2, figsize=(8,3.5))
    else:
        ax, ax2 = axs
        fig = ax.get_figure()

    fig.suptitle(logf.name.as_plot_title())
    ax2.set(
        xlabel='Frequency (GHz)',
        ylabel='phase (rad)',
    )
    ax2.grid()

    freq = logf.df['freq_GHz'].values
    s21_dB = logf.df['s21_mag_dB'].values
    s21_rad = logf.df['s21_phase_rad'].values

    s21_rad_old = np.unwrap(s21_rad)
    s21_rad = misc.remove_e_delay(s21_rad, freq, i_start, i_end)
    ax2.plot(freq, s21_rad_old, '.')
    ax2.plot(freq, s21_rad_old - s21_rad, '-')
    ihalf = int(freq.size/2)
    plotter.cursor(ax2, x=freq[ihalf], text=f'idx={ihalf}', text_style=dict(fontsize='large'))

    s21 = 10 ** (s21_dB / 20) * np.exp(1j*s21_rad)
    cfit = fitter.CurveFit(
        xdata=freq,
        ydata=None,  # fill latter.
        model=models.ResonatorModel_inverse(),
        hold=True,
    )
    s21 = cfit.model.normalize(s21)
    cfit.ydata = 1/s21
    cfit.fit(**kwargs)
    ax = cfit.model.plot(cfit, ax=ax, annotate=annotate, init=init)
    return cfit, ax
    
def fit_coherence(logf, ax=None, model=None, xy=(0.6,0.9), fdata=500, kind=None, **kwargs):
    if ax is None:
        ax = logf.plot1d(ax=ax)

    fig = ax.get_figure()
    fig.set_size_inches(5,3)

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

def plot_xtalk(logf, slope=-0.01, offset=0.0, ax=None, **kwargs):
    """Plot 2d with a guide line. For xtalk data.
    
    Args:
        slope, offset: float, property of the guide line.
        kwargs: passed to logf.plot2d.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4,3))
    else:
        fig = ax.get_figure()
        fig.set_size_inches(4,3)

    logf.plot2d(ax=ax, **kwargs)

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

def plot_ro_mat(logf, return_all=False, plot=True):
    """Plot assignment fidelity matrix along with the labels.
    For data produced by visibility experiment.
    """
    se = logf.df[logf.deps].mean()  # Remove the 'Runs' columns
    n_qs = int(np.sqrt(se.size))
    labels = se.index.values.reshape(n_qs,n_qs)
    ro_mat = se.values.reshape(n_qs,n_qs)

    if plot:
        fig, (ax, ax2) = plt.subplots(ncols=2, figsize=(8,4))
        fig.suptitle(logf.name.as_plot_title())
        ax.set_title('assgiment matrix')
        ax2.set_title('labels')
        plotter.plot_mat2d(ro_mat, ax=ax, fmt=lambda n: f'{n*100:.1f}%')
        plotter.plot_mat2d(np.zeros(labels.shape), labels, ax=ax2)
    else:
        ax, ax2 = None
        
    if return_all:
        return ro_mat, labels, ax, ax2
    else:
        return ro_mat

def plot_tomo_probs(logf, ro_mat=None, n_ops_n_sts=None, return_all=False, figsize=(8,6), plot=True):
    """Plot probabilities after tomo operations along with labels, 
    For data produced by tomo experiment.

    Args:
        n_ops_n_sts: (int, int), shape for the returned matrix.
        if None, inferred from data size.
        ro_mat: matrix (n_sts*n_sts), assgiment fidelity matrix to correct state 
        readout probabilities.
    """
    se = logf.df[logf.deps].mean()
    if n_ops_n_sts is None:
        # Total number of probs should be: (n_ops_1q ** n_qs) * (n_sts_1q ** n_qs)
        n_ops_1q = 3
        n_sts_1q = 2
        n_qs = int(np.log(se.size) / (np.log(n_ops_1q)+np.log(n_sts_1q)))
        n_ops = int(n_ops_1q ** n_qs)
        n_sts = int(n_sts_1q ** n_qs)
    else:
        n_ops, n_sts = n_ops_n_sts

    labels = se.index.values.reshape(n_ops, n_sts)  # State labels runs faster

    probs = se.values.reshape(n_ops, n_sts)
    if ro_mat is not None:
        for i, ps in enumerate(probs):
            probs[i] = np.dot(np.linalg.inv(ro_mat), ps)

    if plot:
        fig, (ax, ax2) = plt.subplots(ncols=2, figsize=figsize)
        fig.suptitle(logf.name.as_plot_title())
        ax.set_title('probs')
        ax2.set_title('labels')
        plotter.plot_mat2d(probs, ax=ax, fmt=lambda n: f'{n*100:.1f}%')
        plotter.plot_mat2d(np.zeros(labels.shape), labels, ax=ax2)
    else:
        ax = None
        ax2 = None

    if return_all:
        return probs, labels, ax, ax2
    else:
        return probs

def plot_qpt(dir, out_ids, in_ids=None, ro_mat_out=None, ro_mat_in=None, plot=True):
    def qst(id, ro_mat):
        lf = fileio.LabradRead(dir, id)
        probs = plot_tomo_probs(lf, ro_mat=ro_mat, plot=False)
        rho = tomo.qst(probs, 'tomo')
        return rho
    if in_ids is None:
        rho_in = {
            '0': np.array([
                [1,0],
                [0,0],
            ]),
            '1': np.array([
                [0,0],
                [0,1],
            ]),
            'x': np.array([
                [.5, .5j],
                [-.5j, .5],
            ]),
            'y': np.array([
                [.5, .5],
                [.5, .5]
            ])
        }
    else:
        rho_in = {k: qst(id, ro_mat_in) for k, id in in_ids.items()}
    
    rho_out = {k: qst(id, ro_mat_out) for k, id in out_ids.items()}

    chi = tomo.qpt(
        [rho_in[k] for k in ('0', 'x', 'y', '1')], 
        [rho_out[k] for k in ('0', 'x', 'y', '1')], 
        'sigma',
    )

    lf = fileio.LabradRead(dir, out_ids['0'])
    if in_ids:
        sid = (f'{min(in_ids.values())}-{max(in_ids.values())}'
            f'-> #{min(out_ids.values())}-{max(out_ids.values())}')
    else:
        sid = f'ideal -> #{min(out_ids.values())}-{max(out_ids.values())}'
    fname = lf.name.as_file_name(id=sid, qubit='')
    ptitle = lf.name.as_plot_title(id=sid, qubit='')
    
    if plot:
        ax_r, ax_i = plotter.plot_complex_mat3d(chi)
        fid = np.abs(chi[0,0])
        ax_r.text2D(0.1,0.9, f'abs($\\chi$), Fidelity={fid*100:.1f}%', transform=ax_r.transAxes)
        ax_r.get_figure().suptitle(ptitle)
    else:
        ax_r = None
        ax_i = None
    
    return chi, rho_in, rho_out, fname, ax_r, ax_i