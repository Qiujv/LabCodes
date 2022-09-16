import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from labcodes import misc, fileio, plotter
import labcodes.frog.pyle_tomo as tomo

JUDGE_TOL = 8


def get_center(conf, qubit, state):
    """Get |0> center or |1> center fron logf.conf, return in a complex number."""
    center = conf['parameter'][f'Device.{qubit.upper()}.|{state}> center']['data'][20:-2].split(', ')
    center = [float(i) for i in center]
    center = center[0] + 1j*center[1]
    return center

def judge(df, conf, qubit='q2', label=None, tolerance=None, return_all=False):
    """Do state discrimination for single shot datas. For example:
        i1, q1 -> cplx_q1, cplx_q1_rot, q1_s1
    
    Adds columns to lf.df. Plots if the 0, 1 center not right for the datas.
    no returns.

    Args:
        df: DataFrame with single Is and Qs.
        conf: lf.conf from which the |0> and |1> center are obtained.
        qubit: str, which qubit to use.
        label: use column i{label}, q{label} as single shot.
            if None, use qubit[1:].
        tolerance: angle in degree. If difference found in angle check larger than this, plot.
    """
    if tolerance is None: tolerance = JUDGE_TOL
    if label is None: label = qubit[1:]
    df = df.copy()

    df[f'cplx_{qubit}'] = df[f'i{label}'] + 1j*df[f'q{label}']
    cent0 = get_center(conf, qubit, 0)
    cent1 = get_center(conf, qubit, 1)

    angle = -np.angle(cent1 - cent0)
    df[f'cplx_{qubit}_rot'] = df[f'cplx_{qubit}'] * np.exp(1j*angle)
    cent0_rot = cent0 * np.exp(1j*angle)
    cent1_rot = cent1 * np.exp(1j*angle)

    thres = (cent0_rot + cent1_rot).real / 2
    mask_1 = df[f'cplx_{qubit}_rot'] > thres
    df[f'{qubit}_s1'] = mask_1

    # Check the 0, 1 center in conf is right.
    _, angle_new = misc.auto_rotate(df[f'cplx_{qubit}'], True)
    angle_diff = (angle - angle_new) % np.pi  # in [0,pi)
    tolerance = tolerance * np.pi/180
    close_enough = (angle_diff <= tolerance) or (np.pi - angle_diff <= tolerance)
    if not close_enough:
        fig, (ax, ax2) = plt.subplots(ncols=2, figsize=(6,3))
        # fig.suptitle(lf.name.as_plot_title(qubit=qubit.upper()))
        plotter.plot_iq(df[f'cplx_{qubit}'], ax=ax)
        ax.plot(cent0.real, cent0.imag, color='C0', marker='*', markeredgecolor='w', markersize=10)
        ax.plot(cent1.real, cent1.imag, color='C1', marker='*', markeredgecolor='w', markersize=10)
        
        plotter.plot_iq(df[f'cplx_{qubit}_rot'][~mask_1], ax=ax2)
        plotter.plot_iq(df[f'cplx_{qubit}_rot'][mask_1], ax=ax2)
        plotter.cursor(ax2, x=round(thres, 4))

        # Plot the angle difference.
        x = np.linspace(*ax2.get_xlim(), 5)[1:-1]
        mean = df[f'cplx_{qubit}_rot'].mean()
        y = mean.imag + np.tan(angle_diff) * (x-mean.real)
        ax2.plot(x,y,'k--')
        ax2.plot(x,np.ones(x.shape)*y[0], 'k-')
        ax2.annotate('{:.1f} deg.'.format(angle_diff * 180/np.pi), (x[1], (y[0]+y[-1])/2), va='center')
    if return_all:
        return df, thres
    else:
        return df


def get_conditional_p1(lf):
    """Get q5 s1_prob condition to q1q2=00, 01, 10, 11 from single shot logfile.
    
    Returns {'00':p1_00,'01':p1_01,'10':p1_10,'11':p1_11}
    """
    df = lf.df.copy()

    if 'q1_s1' in df: df['q1_s1'] = df['q1_s1'].astype(bool)
    else: df = judge(df, lf.conf, 'q1')
    if 'q2_s1' in df: df['q2_s1'] = df['q2_s1'].astype(bool)
    else: df = judge(df, lf.conf, 'q2')
    if 'q5_s1' in df: df['q5_s1'] = df['q5_s1'].astype(bool)
    else: df = judge(df, lf.conf, 'q5')

    df = df[['runs', 'q1_s1', 'q2_s1', 'q5_s1']]
    masks = {
        '00': (~df['q1_s1']) & (~df['q2_s1']),
        '01': (~df['q1_s1']) & ( df['q2_s1']),
        '10': ( df['q1_s1']) & (~df['q2_s1']),
        '11': ( df['q1_s1']) & ( df['q2_s1']),
        'all': np.ones(df.shape[0], dtype=bool),
    }
    # Appearance probability, q5_s1_prob of certain select.
    df_out = {k: [mask.mean(), df.loc[mask, 'q5_s1'].mean(), 1-df.loc[mask, 'q5_s1'].mean()]
          for k, mask in masks.items()}
    df_out = pd.DataFrame.from_dict(df_out, orient='index', columns=['weight', 'q5_s1_prob', 'q5_s0_prob'])
    df_out.fillna(0)  # np.mean returns nan if array is [], change that results to 0.
    return df_out

def single_shot_qst(dir, id0, idx, idy, select, ro_mat=None, suffix='csv'):
    """Calculate density matrix from single shot tomo experiments, with tomo op: I, X/2, Y/2.
    
    Args:
        dir: directory where the logfiles are.
        id0, idx2, idy2: int, id of logfiles for tomo experiments.
        select: conditional state, key of dict returned by get_conditional_p1.
        ro_mat: np.array, readout assignment matrix of q5. 
            if None, apply I.
    """
    dfs = [get_conditional_p1(fileio.LabradRead(dir, id, suffix=suffix)) for id in (id0, idx, idy)]
    probs = [[df.loc[select, 'q5_s0_prob'], df.loc[select, 'q5_s1_prob']] for df in dfs]

    if ro_mat is not None:
        for i, ps in enumerate(probs):
            probs[i] = np.dot(np.linalg.inv(ro_mat), ps)

    rho = tomo.qst(np.array(probs), 'tomo')
    return rho

def single_shot_qpt(dir, m, selects=('00','01','10','11'), verbose=True, **kw):
    """Calculate process matrix from single shot tomo experiments, with:
        init_state x tomo_op: (0, X, Y, 1) x (0, X, Y) = 00, 0X, ...
        12 logfiles with id from m on.
    
    Returns: 
        chi_all: {'00': chi}, chi is 4x4 complex matrix.
        Fchi: {'00': Fid(chi, chi_ideal)}.
        Frho: {'00': [Fid(rho_0), Fid(rho_x), Fid(rho_y), Fid(rho_1)]}.
        chi_ideal_all: {'00': chi_ideal}.
    """
    chi_all = {}
    chi_ideal_all = {}
    Fchi = {}
    Frho = {}
    for select in selects:
        rho_out = {
            '0': single_shot_qst(dir, m+0, m+1, m+2, select, **kw),
            'x': single_shot_qst(dir, m+3, m+4, m+5, select, **kw),
            'y': single_shot_qst(dir, m+6, m+7, m+8, select, **kw),
            '1': single_shot_qst(dir, m+9, m+10, m+11, select, **kw),
        }
        rho_out_ideal = {
            '0': np.array(rho_Q5(select, 1,0)),
            'x': np.array(rho_Q5(select, 1/np.sqrt(2),-1j/np.sqrt(2))),
            'y': np.array(rho_Q5(select, 1/np.sqrt(2),1/np.sqrt(2))),
            '1': np.array(rho_Q5(select, 0,1)),
        }
    
        # plotter.plot_complex_mat3d(rho_out['0'])
        # plotter.plot_complex_mat3d(rho_out_ideal['0'])
        rho_f_0 = fidelity(rho_out['0'], rho_out_ideal['0'])
        # plotter.plot_complex_mat3d(rho_out['x'])
        # plotter.plot_complex_mat3d(rho_out_ideal['x'])
        rho_f_x = fidelity(rho_out['x'], rho_out_ideal['x'])
        # plotter.plot_complex_mat3d(rho_out['y'])
        # plotter.plot_complex_mat3d(rho_out_ideal['y'])
        rho_f_y = fidelity(rho_out['y'], rho_out_ideal['y'])
        # plotter.plot_complex_mat3d(rho_out['1'])
        # plotter.plot_complex_mat3d(rho_out_ideal['1'])
        rho_f_1 = fidelity(rho_out['1'], rho_out_ideal['1'])
        Frho[select] = [rho_f_0, rho_f_x, rho_f_y, rho_f_1]

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
        chi = tomo.qpt(
            [rho_in[k] for k in ('0', 'x', 'y', '1')], 
            [rho_out[k] for k in ('0', 'x', 'y', '1')], 
            'sigma',
        )

        chi_ideal = tomo.qpt(
            [rho_in[k] for k in ('0', 'x', 'y', '1')], 
            [rho_out_ideal[k] for k in ('0', 'x', 'y', '1')], 
            'sigma',
        )

        # chi_fidelity = fidelity(chi, chi_ideal)
        chi_fidelity = np.abs(chi).max()
        chi_all[select] = chi
        chi_ideal_all[select] = chi_ideal
        Fchi[select] = chi_fidelity

    if verbose:
        # for select, f in Fchi.items():
        #     print(f'select {select}: F={f:.3f}')

        ax = plotter.plot_mat2d(pd.DataFrame(Frho), fmt='{:.3f}'.format)
        ax.set(
            title='rho_fidelitys',
            xlabel='select',
            xticklabels=['', '00', '01', '10', '11'],
            ylabel='state',
            yticklabels=['', '0', 'X', 'Y', '1'],
        )

        fig, _ = plot_chi_all(chi_all, f'tele ss qpt, m={m}')
        # plot_chi_all(chi_ideal_all, f'tele ss qpt, m={m}')
    else:
        fig = None

    return chi_all, Fchi, Frho, chi_ideal_all, fig

def plot_chi_all(chi_all, figtitle=None):
    fig = plt.figure(figsize=(14,8), tight_layout=False)
    axs = [fig.add_subplot(2,4,i, projection='3d') for i in range(1,9)]
    iis = {'00': (0,1), '01': (2,3), '10': (4,5), '11': (6,7)}
    colorbar = True  # Only for the first plot
    for select, chi in chi_all.items():
        ii, sii = iis[select]
        plotter.plot_complex_mat3d(chi, axs[ii:sii+1], cmin=-1, cmax=1, colorbar=colorbar)
        colorbar = False
        axs[sii].set_title(f'select={select}')
    if figtitle: fig.suptitle(figtitle)
    return fig, axs

def rho_Q5(q1q2s, alpha, beta):
    """Returns theoritical density matrix of Q5 after teleport.
    
    Args:
        q1q2s: '00', '01', '10', '11'.
        alpha, bete: float, coefficient of state |0> and |1>.
    """
    base_vector = {
        '0': np.array([
            [1],
            [0],
        ]),
        '1': np.array([
            [0],
            [1],
        ])
    }
    if q1q2s == '00': 
        q3s = alpha*base_vector['0'] + beta*base_vector['1']
    elif q1q2s == '01': 
        q3s = alpha*base_vector['0'] + beta*base_vector['1']
    elif q1q2s == '10': 
        q3s = alpha*base_vector['0'] + beta*base_vector['1']
    elif q1q2s == '11':
        q3s = alpha*base_vector['0'] + beta*base_vector['1']
    else:
        raise ValueError(q1q2s)

    q3s = np.matrix(q3s)
    return np.dot(q3s, q3s.H)

def fidelity(rho, sigma):
    return np.real(np.trace(np.dot(rho, sigma)))

