"""Script provides functions dealing with routine experiment datas."""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
import labcodes.frog.pyle_tomo as tomo
from labcodes import fileio, fitter, misc, models, plotter
from labcodes.frog import tele


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
    
def fit_coherence(logf, ax=None, model=None, kind=None, xmax=None, **kwargs):
    if ax is None:
        ax = logf.plot1d(ax=ax, y_name='s1_prob')

    fig = ax.get_figure()
    fig.set_size_inches(5,3)

    if kind is None: kind = str(logf.name)
    if 'T1' in kind:
        mod = models.ExponentialModel()
        symbol = 'T_1'
    elif ('Ramsey' in kind) or ('T2' in kind):
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

    if xmax is None:
        mask = np.ones(logf.df.shape[0], dtype='bool')
    else:
        mask = logf.df[xname].values <= xmax

    cfit = fitter.CurveFit(
        xdata=logf.df[xname].values[mask],
        ydata=logf.df['s1_prob'].values[mask],
        model=mod,
        hold=True,
    )
    cfit.fit(**kwargs)

    fdata = np.linspace(logf.df[xname].min(), logf.df[xname].max(), 5*logf.df.shape[0])
    ax.plot(*cfit.fdata(fdata), 'r-', lw=1)
    ax.annotate(f'${symbol}\\approx {cfit["tau"]:,.2f}\\pm{cfit["tau_err"]:,.4f} {xname[-2:]}$', 
        (0.6,0.9), xycoords='axes fraction')
    # ax.annotate(f'offset={cfit["offset"]:.2f}$\\pm${cfit["offset_err"]:.2f}', 
    #     (0.95,0.1), xycoords='axes fraction', ha='right')
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

    Note: **slope = - xtalk**.
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
    ax.annotate(f'{-slope*100:.2f}%', c, size='xx-large', ha='left', va='bottom', 
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

def plot_ro_mat(logf, ax=None, return_all=False):
    """Plot assignment fidelity matrix along with the labels.
    For data produced by visibility experiment.
    """
    se = logf.df[logf.deps].mean()  # Remove the 'Runs' columns
    n_qs = int(np.sqrt(se.size))
    labels = se.index.values.reshape(n_qs,n_qs).T  # Transpose to assignment matrix we usually use. Check Carefully.
    ro_mat = se.values.reshape(n_qs,n_qs).T

    if ax:
        ax.set_title(logf.name.as_plot_title())
        plotter.plot_mat2d(ro_mat, ax=ax, fmt=lambda n: f'{n*100:.1f}%')
        print('Matrix labels:\n', labels)

    if return_all:
        return ro_mat, labels, ax
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

def plot_qst(dir, id, ro_mat=None, fid=None, normalize=False):
    lf = fileio.LabradRead(dir, id)
    probs = plot_tomo_probs(lf, ro_mat=ro_mat, plot=False)
    n_ops_1q = 3
    n_sts_1q = 2
    n_qs = int(np.log(probs.size) / (np.log(n_ops_1q)+np.log(n_sts_1q)))

    if n_qs == 1:
        rho = tomo.qst(probs, 'tomo')
    else:
        rho = tomo.qst(probs, f'tomo{n_qs}')
    labels = misc.bitstrings(n_qs)
    rho_abs = np.abs(rho)
    if normalize is True: rho = rho / np.trace(rho_abs)
    ax = plotter.plot_mat3d(rho_abs)

    if fid is None:
        # Calculate fidelity with guessed state.
        rho_sort = np.sort(rho_abs.ravel())
        if rho_sort[-1]-rho_sort[-4] > 0.3:
            # Guess it is a simple state.
            fid = rho_sort[-1]
            msg = 'highest bar'
        else:
            # Guess it is a bipartite entangle state.
            fid = np.sum(rho_sort[-4:]) / 2
            msg = 'sum(highest bars)/2'
    else:
        msg = 'Fidelity'
    ax.text2D(0.0,0.9, f'abs($\\rho$), {msg}={fid*100:.1f}%', 
        transform=ax.transAxes, fontsize='x-large')
    cbar = ax.collections[0].colorbar
    cbar.set_label('$|\\rho|$')
    ax.set(
        title=lf.name.as_plot_title(),
        xticklabels=labels,
        yticklabels=labels,
    )
    fname = lf.name.as_file_name()
    return rho, fname, ax

def plot_qpt(dir, out_ids, in_ids=None, ro_mat_out=None, ro_mat_in=None):
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
    rho_out['0']

    chi = tomo.qpt(
        [rho_in[k] for k in ('0', 'x', 'y', '1')], 
        [rho_out[k] for k in ('0', 'x', 'y', '1')], 
        'sigma',
    )

    # Resolve plot titles.
    lf = fileio.LabradRead(dir, out_ids['0'])
    if in_ids:
        sid = (f'{min(in_ids.values())}-{max(in_ids.values())}'
            f'-> #{min(out_ids.values())}-{max(out_ids.values())}')
    else:
        sid = f'{min(out_ids.values())}-{max(out_ids.values())} <- ideal'
    fname = lf.name.as_file_name(id=sid)
    ptitle = lf.name.as_plot_title(id=sid)

    ax = plotter.plot_mat3d(np.abs(chi))
    fid = np.abs(chi[0,0])
    ax.text2D(0.0,0.9, f'abs($\\chi$), Fidelity={fid*100:.1f}%', 
        transform=ax.transAxes, fontsize='x-large')
    cbar = ax.collections[0].colorbar
    cbar.set_label('$|\\chi|$')
    ax.get_figure().suptitle(ptitle)

    return chi, rho_in, rho_out, fname, ax
    


def df2mat(df, fname=None, xy_name=None):
    """Save dataframe to .mat file. For internal communication.
    
    Args:
        df: DataFrame to save.
        fname: name of file to save.
        xy_names: (x_name, y_name), if given, reshape df into 2d array before saving.
    """
    if xy_name is None:
        mdic = df.to_dict('list')
    else:
        x_name, y_name = xy_name
        df = df.sort_values(by=[x_name, y_name])
        xuni = df[x_name].unique()
        xsize = xuni.size
        ysize = df.shape[0] // xsize
        mdic = {col: df[col].values.reshape(xsize, ysize)
                for col in df.columns}

    if fname: scipy.io.savemat(fname, mdic)
    return mdic


from scipy import stats
from scipy.optimize import leastsq


def rb_fit(dir, id, id_ref, residue=None):
    """Adapted from codes provided by NJJ, but remove dependence to labrad."""
    lf = fileio.LabradRead(dir, id)
    df2d = df2mat(lf.df, xy_name=['k', 'm'])
    mat = df2d['prob_s0']
    m = df2d['m'][0]

    lf0 = fileio.LabradRead(dir, id_ref)
    df2d0 = df2mat(lf0.df, xy_name=['k', 'm'])
    mat0 = df2d0['prob_s0']
    m0 = df2d0['m'][0]
    gate = lf.conf['parameter']['gate']['data'][1:-1]

    prob = np.nanmean(mat,axis=0)
    prob_std = np.nanstd(mat,axis=0)
    prob0 = np.nanmean(mat0,axis=0)
    prob0_std = np.nanstd(mat0,axis=0)
    if residue is None:
        p0 = np.array([0.5,0.95,0.5])
        def fitfunc(p,t):
            return p[0]*p[1]**t+p[2]
    else:
        p0 = np.array([0.5,0.95])
        def fitfunc(p,t):
            return p[0]*p[1]**t+residue
    def errfunc(p):
        return fitfunc(p,m) - prob
    out = leastsq(errfunc,p0,full_output=True)
    p = out[0]
    pgate = p[1]
    vdsig = stats.norm.fit(errfunc(p))[1] # the sigma of the residue.
    var = np.sqrt(out[1][1,1])*np.abs(vdsig)

    def errfunc0(p):
        return fitfunc(p,m0) - prob0
    out0 = leastsq(errfunc0,p0,full_output=True)
    p0 = out0[0]
    pr = p0[1]
    vdsig0 = stats.norm.fit(errfunc0(p))[1] # the sigma of the residue.
    var0 = np.sqrt(out[1][1,1])*np.abs(vdsig0)

    if residue is not None:
        p0 = np.concatenate([p0,[residue]])
        p = np.concatenate([p,[residue]])
    rgate = (1-pgate/pr)/2.0
    rstd = 0.5*(pgate**2/pr**2*(var**2/pgate**2+var0**2/pr**2))**0.5
    print('pgate:%.4f\npr:%.4f'%(pgate,pr))
    print('gate error=%.4f+-%.4f'%(rgate,rstd))
    fig, ax = plt.subplots()
    ax.errorbar(m,prob,yerr=prob_std,fmt='rs',label=gate+' gate',alpha=0.8,ms=3)
    ax.plot(m,fitfunc(p,m),'r--')
    ax.errorbar(m0,prob0,yerr=prob0_std,fmt='bo',label='reference',alpha=0.8,ms=3)
    ax.plot(m0,fitfunc(p0,m0),'b--')
    plt.xlabel('m - Number of Gates')
    plt.ylabel('Sequence Fidelity')
    plt.ylim([0.5,1])
    # plt.xlim([min(m),max(m)])
    plt.grid(True,which='both')
    plt.title(f'id={id}, ref={id_ref}, {lf.name.qubit.lower()}, '+gate+r' gate fidelity $%.2f\pm%.2f$%%'%(100-rgate*100,100*rstd))

    ax.text(0.5, 0.75, r'$%.4f\times %.4f^m+%.4f$'%(p0[0],p0[1],p0[2]),
        horizontalalignment='left',
        verticalalignment='center',
        fontsize=12, color='b',
        transform=ax.transAxes)
    ax.text(0.5, 0.65, r'$%.4f\times %.4f^m+%.4f$'%(p[0],p[1],p[2]),
        horizontalalignment='left',
        verticalalignment='center',
        fontsize=12, color='r',
        transform=ax.transAxes)
    ax.legend()
    return ax

def plot_rb(dir, id, id_ref, residue=None):
    lf = fileio.LabradRead(dir, id)
    lf0 = fileio.LabradRead(dir, id_ref)
    gate = lf.conf['parameter']['gate']['data'][1:-1]
    lf_name = lf.name.copy()
    lf_name.id = f'{lf.name.id} ref {lf0.name.id}'
    df = pd.concat([
        lf.df.groupby(by='m').mean()[['prob_s0', 'prob_s1']],
        lf.df.groupby(by='m').std()[['prob_s0', 'prob_s1']
            ].rename(columns={'prob_s0': 'prob_s0_std', 'prob_s1': 'prob_s1_std'}),
    ], axis=1).reset_index()
    df0 = pd.concat([
        lf0.df.groupby(by='m').mean()[['prob_s0', 'prob_s1']],
        lf0.df.groupby(by='m').std()[['prob_s0', 'prob_s1']
            ].rename(columns={'prob_s0': 'prob_s0_std', 'prob_s1': 'prob_s1_std'}),
    ], axis=1).reset_index()

    def rb_decay(x, amp=0.5, fid=0.99, residue=0.5):
        return amp * fid**x + residue

    mod = models.MyModel(rb_decay)
    if residue: mod.set_param_hint('residue', vary=False, value=residue)

    cfit = fitter.CurveFit(
        xdata=df['m'].values,
        ydata=df['prob_s0'].values,
        model=mod,
    )
    cfit0 = fitter.CurveFit(
        xdata=df0['m'].values,
        ydata=df0['prob_s0'].values,
        model=mod,
    )
    gate_err = (1 - cfit['fid']/cfit0['fid']) / 2
    gate_err_std = 0.5 * cfit['fid']*cfit0['fid'] \
                * abs(cfit['fid_err']/cfit['fid'] + 1j*cfit0['fid_err']/cfit0['fid'])
    lf_name.title = lf_name.title.replace('Randomized Benchmarking', 'RB')\
                    + f'gate fidelity {(1-gate_err)*100:.2f}%±{gate_err_std*100:.3f}%'

    fig, ax = plt.subplots()
    ax.errorbar('m', 'prob_s0', 'prob_s0_std', data=df, fmt='rs', label=f'{gate} gate', 
                alpha=0.8, markersize=3)
    ax.errorbar('m', 'prob_s0', 'prob_s0_std', data=df0, fmt='bo', label='reference', 
                alpha=0.8, markersize=3)
    ax.plot(*cfit.fdata(500), 'r--')
    ax.plot(*cfit0.fdata(500), 'b--')
    ax.annotate('${:.4f}\\times{:.4f}^m + {:.4f}$'.format(cfit['amp'], cfit['fid'], cfit['residue']), 
                (1.0, 0.76), ha='right', va='center', color='b', xycoords='axes fraction')
    ax.annotate('${:.4f}\\times{:.4f}^m + {:.4f}$'.format(cfit0['amp'], cfit0['fid'], cfit0['residue']), 
                (1.0, 0.69), ha='right', va='center', color='r', xycoords='axes fraction')
    ax.grid(True)
    ax.set(
        title=lf_name.as_plot_title(),
        xlabel='m - Number of Gates',
        xlim=(0, df['m'].max()+10),
        ylabel='Sequence Fidelity',
        ylim=(0.5,1),
    )
    ax.legend()
    return ax, lf_name

def plot_rb_multi(dir, ids, id_ref, residue=None):
    lfs = [fileio.LabradRead(dir, id) for id in ids]
    lf0 = fileio.LabradRead(dir, id_ref)
    gates = [lf.conf['parameter']['gate']['data'][1:-1] for lf in lfs]
    lf_name = lfs[0].name.copy()
    lf_name.id = ', '.join([str(lf.name.id) for lf in lfs] + [f'ref {lf0.name.id}'])
    dfs = [pd.concat([
        lf.df.groupby(by='m').mean()[['prob_s0', 'prob_s1']],
        lf.df.groupby(by='m').std()[['prob_s0', 'prob_s1']
            ].rename(columns={'prob_s0': 'prob_s0_std', 'prob_s1': 'prob_s1_std'}),
    ], axis=1).reset_index() for lf in lfs]
    df0 = pd.concat([
        lf0.df.groupby(by='m').mean()[['prob_s0', 'prob_s1']],
        lf0.df.groupby(by='m').std()[['prob_s0', 'prob_s1']
            ].rename(columns={'prob_s0': 'prob_s0_std', 'prob_s1': 'prob_s1_std'}),
    ], axis=1).reset_index()

    def rb_decay(x, amp=0.5, fid=0.99, residue=0.5):
        return amp * fid**x + residue

    mod = models.MyModel(rb_decay)
    if residue: mod.set_param_hint('residue', vary=False, value=residue)

    cfits = [fitter.CurveFit(
        xdata=df['m'].values,
        ydata=df['prob_s0'].values,
        model=mod,
    ) for df in dfs]
    cfit0 = fitter.CurveFit(
        xdata=df0['m'].values,
        ydata=df0['prob_s0'].values,
        model=mod,
    )
    gates_err = [(1 - cfit['fid']/cfit0['fid']) / 2 for cfit in cfits]
    gates_err_std = [0.5 * cfit['fid']*cfit0['fid'] \
                    * abs(cfit['fid_err']/cfit['fid'] + 1j*cfit0['fid_err']/cfit0['fid'])
                    for cfit in cfits]
    lf_name.title = lf_name.title.replace(' Randomized Benchmarking', 'RB'
                    ).replace(gates[0], '')\
                    + f'ave fidelity {(1-np.mean(gates_err))*100:.2f}%'

    fig, ax = plt.subplots()
    [ax.errorbar('m', 'prob_s0', 'prob_s0_std', data=df, fmt='s', label=f'{gate} {(1-gate_err)*100:.2f}%±{gate_err_std*100:.3f}%', 
                alpha=0.8, markersize=3)
        for df, gate, gate_err, gate_err_std in zip(dfs, gates, gates_err, gates_err_std)]
    ax.errorbar('m', 'prob_s0', 'prob_s0_std', data=df0, fmt='ko', label='reference', 
                alpha=0.8, markersize=3)
    ax.set_prop_cycle(None)
    [ax.plot(*cfit.fdata(500), '--') for cfit in cfits]
    ax.plot(*cfit0.fdata(500), 'k--')

    ax.grid(True)
    ax.set(
        title=lf_name.as_plot_title(),
        xlabel='m - Number of Gates',
        xlim=(0, max([df['m'].max() for df in dfs])+10),
        ylabel='Sequence Fidelity',
        ylim=(0.5,1),
    )
    ax.legend()
    return ax, lf_name

def plot_iq_2q(dir, id00, id01=None, id10=None, id11=None):
    """Plot two qubit joint readout IQ scatter, for single_shot_2q."""
    def load_one(lf, qb):
        df, thres = tele.judge(lf.df, lf.conf, qubit=qb, return_all=True, tolerance=np.inf)
        df = df[[f'cplx_{qb}_rot', f'{qb}_s1']
            ].rename(columns={f'cplx_{qb}_rot': 'cplx_rot', f'{qb}_s1': 's1'})
        return df, thres  # thres get from experiment parameters.

    if id01 is None: id01 = id00 + 1
    if id10 is None: id10 = id00 + 2
    if id11 is None: id11 = id00 + 3

    lf00 = fileio.LabradRead(dir, id00, suffix='csv_complete')
    df00q1, thres1 = load_one(lf00, 'q1')
    df00q2, thres2 = load_one(lf00, 'q2')
    lf01 = fileio.LabradRead(dir, id01, suffix='csv_complete')
    df01q1, _ = load_one(lf01, 'q1')
    df01q2, _ = load_one(lf01, 'q2')
    lf10 = fileio.LabradRead(dir, id10, suffix='csv_complete')
    df10q1, _ = load_one(lf10, 'q1')
    df10q2, _ = load_one(lf10, 'q2')
    lf11 = fileio.LabradRead(dir, id11, suffix='csv_complete')
    df11q1, _ = load_one(lf11, 'q1')
    df11q2, _ = load_one(lf11, 'q2')

    lf_names = lf00.name.copy()
    lf_names.id = ','.join([str(lf.name.id) for lf in [lf00, lf01, lf10, lf11]])
    df = pd.DataFrame({
        'c00': df00q1['cplx_rot'].values.real + 1j*df00q2['cplx_rot'].values.real,
        'c01': df01q1['cplx_rot'].values.real + 1j*df01q2['cplx_rot'].values.real,
        'c10': df10q1['cplx_rot'].values.real + 1j*df10q2['cplx_rot'].values.real,
        'c11': df11q1['cplx_rot'].values.real + 1j*df11q2['cplx_rot'].values.real,
        's00': (~df00q1['s1'].values) & (~df00q2['s1'].values),
        's01': (~df01q1['s1'].values) & ( df01q2['s1'].values),
        's10': ( df10q1['s1'].values) & (~df10q2['s1'].values),
        's11': ( df11q1['s1'].values) & ( df11q2['s1'].values),
    })

    fig, ax = plt.subplots()
    plotter.plot_iq(df['c00'], ax=ax, label='|00>')
    plotter.plot_iq(df['c01'], ax=ax, label='|01>')
    plotter.plot_iq(df['c10'], ax=ax, label='|10>')
    plotter.plot_iq(df['c11'], ax=ax, label='|11>')
    ax.annotate(f'p00_s00={df["s00"].mean():.3f}', (0.05,0.05), xycoords='axes fraction')
    ax.annotate(f'p01_s01={df["s01"].mean():.3f}', (0.05,0.95), xycoords='axes fraction')
    ax.annotate(f'p10_s10={df["s10"].mean():.3f}', (0.55,0.05), xycoords='axes fraction')
    ax.annotate(f'p11_s11={df["s11"].mean():.3f}', (0.55,0.95), xycoords='axes fraction')
    ax.axvline(x=thres1, color='k', ls='--')
    ax.axhline(y=thres2, color='k', ls='--')
    ax.legend(bbox_to_anchor=(1,1))
    ax.tick_params(direction='in')
    ax.set(
        title=lf_names.as_plot_title(),
        xlabel='Q1 projection position',
        ylabel='Q2 projection position',
    )

    return ax, lf_names

def plot_2q_qpt(dir, start, ro_mat=None, plot_all=False):
    """Process two-qubit QPT datas.
    
    Log files of state tomography with prepared state: [0,x,y,1]**2 = 00, 0x, ... 11 (16 in total).
    starts from `start` in `dir`.
    """
    rho_out = []
    for i in np.arange(16)+start:
        rho, _, ax = plot_qst(dir, i, ro_mat=ro_mat)
        ax.set_zlim(0,1)
        rho_out += [rho]
        if not plot_all: plt.close(ax.get_figure())

    rho_1q = [
        np.array([
            [1,0],
            [0,0],
        ]),
        np.array([
            [.5, .5j],
            [-.5j, .5],
        ]),
        np.array([
            [.5, .5],
            [.5, .5]
        ]),
        np.array([
            [0,0],
            [0,1],
        ]),
    ]
    rho_in = tomo.tensor_combinations(rho_1q, 2)

    cz = np.diag([1,1,1,-1])
    rho_ideal = [np.dot(np.dot(cz, x), cz.conj().transpose()) for x in rho_in]
    chi_ideal = tomo.qpt(rho_in, rho_ideal, 'sigma2')
    chi_out = tomo.qpt(rho_in, rho_out, 'sigma2')

    ax, _ = plotter.plot_complex_mat3d(chi_out, label=False)

    fid = tele.fidelity(chi_ideal, chi_out)
    lf_name = fileio.LabradRead(dir, start).name
    lf_name.id = f'{start}-{start+15}'
    lf_name.title += f', F={fid*100:.2f}%'
    ax.get_figure().suptitle(lf_name.as_plot_title())

    return ax, lf_name