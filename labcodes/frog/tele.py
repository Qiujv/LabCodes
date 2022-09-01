import numpy as np
import matplotlib.pyplot as plt
from labcodes import misc, fileio, plotter
import labcodes.frog.pyle_tomo as tomo


def get_center(conf, qubit, state):
    """Get |0> center or |1> center fron logf.conf, return in a complex number."""
    center = conf['parameter'][f'Device.{qubit.upper()}.|{state}> center']['data'][20:-2].split(', ')
    center = [float(i) for i in center]
    center = center[0] + 1j*center[1]
    return center

def judge(lf, qubit='q2', label=None, tolerance=8):
    """Do state discrimination for single shot datas. For example:
        i1, q1 -> cplx_q1, cplx_q1_rot, q1_s1
    
    Adds columns to lf.df. Plots if the 0, 1 center not right for the datas.
    no returns.

    Args:
        lf: logfile with df and conf.
        qubit: str, which qubit to use.
        label: use column i{label}, q{label} as single shot.
            if None, use qubit[1:].
        tolerance: angle in degree. If difference found in angle check larger than this, plot.
    """
    if label is None: label = qubit[1:]
    df = lf.df

    df[f'cplx_{qubit}'] = df[f'i{label}'] + 1j*df[f'q{label}']
    cent0 = get_center(lf.conf, qubit, 0)
    cent1 = get_center(lf.conf, qubit, 1)

    angle = -np.angle(cent1 - cent0)
    df[f'cplx_{qubit}_rot'] = df[f'cplx_{qubit}'] * np.exp(1j*angle)
    cent0_rot = cent0 * np.exp(1j*angle)
    cent1_rot = cent1 * np.exp(1j*angle)

    thres = (cent0_rot + cent1_rot).real / 2
    mask_1 = df[f'cplx_{qubit}_rot'] > thres
    df[f'{qubit}_s1'] = mask_1

    # Check the 0, 1 center in conf is right.
    _, angle_indept = misc.auto_rotate(df[f'cplx_{qubit}'], True)
    angle_diff = (angle - angle_indept) % np.pi  # in [0,pi)
    tolerance = tolerance * np.pi/180
    close_enough = (angle_diff <= tolerance) or (np.pi - angle_diff <= tolerance)
    if not close_enough:
        fig, (ax, ax2) = plt.subplots(ncols=2, figsize=(6,3))
        fig.suptitle(lf.name.as_plot_title(qubit=qubit))
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


def get_conditional_p1(lf):
    """Get q5 s1_prob condition to q1q2=00, 01, 10, 11 from single shot logfile.
    
    Adds columns to lf.df. Returns {'00':p1_00,'01':p1_01,'10':p1_10,'11':p1_11}
    """
    judge(lf, 'q1')
    judge(lf, 'q2')
    judge(lf, 'q5')
    df = lf.df[['runs', 'q1_s1', 'q2_s1', 'q5_s1']]
    p1_00 = df.loc[(~df['q1_s1']) & (~df['q2_s1']), 'q5_s1'].mean()
    p1_01 = df.loc[(~df['q1_s1']) & ( df['q2_s1']), 'q5_s1'].mean()
    p1_10 = df.loc[( df['q1_s1']) & (~df['q2_s1']), 'q5_s1'].mean()
    p1_11 = df.loc[( df['q1_s1']) & ( df['q2_s1']), 'q5_s1'].mean()
    probs = {'00':p1_00,'01':p1_01,'10':p1_10,'11':p1_11}
    return probs

def single_shot_qst(dir, id0, idx2, idy2, select, ro_mat=None):
    """Calculate density matrix from single shot tomo experiments, with tomo op: I, X/2, Y/2.
    
    Args:
        dir: directory where the logfiles are.
        id0, idx2, idy2: int, id of logfiles for tomo experiments.
        select: conditional state, key of dict returned by get_conditional_p1.
        ro_mat: np.array, readout assignment matrix of q5. 
            if None, apply I.
    """
    probs = [get_conditional_p1(fileio.LabradRead(dir, id))[select] 
             for id in (id0, idx2, idy2)]
    probs = [[1-p1, p1] for p1 in probs]

    if ro_mat is not None:
        for i, ps in enumerate(probs):
            probs[i] = np.dot(np.linalg.inv(ro_mat), ps)

    rho = tomo.qst(np.array(probs), 'tomo')
    return rho

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