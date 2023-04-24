"""Process single shot tomo datas, for state teleport experiments."""

from functools import cache
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import labcodes.frog.pyle_tomo as tomo
from labcodes import fileio, misc, plotter


def rho(alpha, beta):
    """Returns density matrix of qubit state alpha*|0> + beta*|1>"""
    state = alpha*np.array([[1],[0]]) + beta*np.array([[0],[1]])
    state = np.matrix(state)
    return np.array(np.dot(state, state.H))

def fidelity(rho, sigma):
    # return np.real(np.trace(np.dot(rho, sigma)))
    return np.real(np.trace(rho @ sigma))  # Preferred according to 

class single_shot_data:
    """Load a single shot data lf, judge it with 0, 1 center stored in lf.conf.
    then produce probs of p_qubits conditional to c_qubits.

    Check .df for original data, check .probs for probilities table.
    """
    def __init__(self, folder, id, suffix='csv', c_qubits=('q1',), p_qubits=('q2',)):
        lf = fileio.LabradRead(folder, id, suffix=suffix)
        self.lf = lf

        if isinstance(c_qubits, str):
            self.c_qubits = [c_qubits]
        else:
            self.c_qubits = [qn.lower() for qn in c_qubits]
        if isinstance(p_qubits, str):
            self.p_qubits = [p_qubits]
        else:
            self.p_qubits = [qn.lower() for qn in p_qubits]
        self.qubits = self.c_qubits + self.p_qubits

        df = lf.df.copy()
        try:
            self.construct()
        except Exception as exc:
            print('WARNING: single_shot_data.construct fails\n'
                  f'Error reports {repr(exc)}')
            self.df = None
            self.probs = None

    def construct(self, tolerance=None):
        df = self.lf.df.copy()
        for qn in self.qubits:
            if f'{qn}_s1' in df:
                df[f'{qn}_s1'] = df[f'{qn}_s1'].astype(bool)
            else:
                df = self.judge(df, qn, drop=True)
                if tolerance is not None:
                    if not self.close_enouth(qn, tolerance=tolerance):
                        self.plot_1q(qn)
        self.df = df
        self.probs = self.get_probs()

    def get_center(self, qubit, state):
        """Get |0> center or |1> center fron logf.conf, return in a complex number."""
        center = self.lf.conf['parameter'][f'Device.{qubit.upper()}.|{state}> center'][20:-2].split(', ')
        center = [float(i) for i in center]
        center = center[0] + 1j*center[1]
        return center

    def get_ang_thres(self, qubit):
        cent0 = self.get_center(qubit, 0)
        cent1 = self.get_center(qubit, 1)
        angle = -np.angle(cent1 - cent0)

        cent0_rot = cent0 * np.exp(1j*angle)
        cent1_rot = cent1 * np.exp(1j*angle)
        thres = (cent0_rot + cent1_rot).real / 2
        return angle, thres

    def judge(self, df, qubit, label=None, drop=False):
        """Do state discrimination for single shot datas. For example:
            i1, q1 -> cplx_q1, cplx_q1_rot, q1_s1.

        Returns new df with added columns.
        """
        if label is None: label = qubit[1:]

        angle, thres = self.get_ang_thres(qubit)
        df[f'cplx_{qubit}'] = df[f'i{label}'] + 1j*df[f'q{label}']
        df[f'cplx_{qubit}_rot'] = df[f'cplx_{qubit}'] * np.exp(1j*angle)
        df[f'{qubit}_s1'] = df[f'cplx_{qubit}_rot'] > thres

        if drop: df.drop(columns=[f'i{label}', f'q{label}'], inplace=True)
        return df

    def get_probs(self, c_qubits=None, p_qubits=None, df=None):
        """df[q1_s1, q2_s1, q3_s1] 
        ---c12, p3---> 
        df: columns=[p0, p1], index=[c00, c01, c10, c11].
        """
        if df is None: df = self.df
        if c_qubits is None: c_qubits = self.c_qubits
        if p_qubits is None: p_qubits = self.p_qubits

        c_states = [''.join(i) for i in product('01', repeat=len(c_qubits))]
        p_states = [''.join(i) for i in product('01', repeat=len(p_qubits))]
        c_prefix = ''.join(c_qubits)
        p_prefix = ''.join(p_qubits)
        probs = {}
        for c_state in c_states:
            probs[f'{c_prefix}_{c_state}'] = {}
            for p_state in p_states:
                mask = np.ones(df.shape[0], dtype=bool)
                for qn, state in zip(c_qubits+p_qubits, c_state+p_state):
                    if state == '0':
                        mask = mask & (~df[f'{qn}_s1'])
                    else:
                        mask = mask & ( df[f'{qn}_s1'])
                probs[f'{c_prefix}_{c_state}'][f'{p_prefix}_{p_state}'] = mask.mean()
        probs = pd.DataFrame.from_records(probs).T
        c_weights = probs.sum(axis='columns')
        probs = probs.divide(c_weights, axis='index')
        probs['weight'] = c_weights
        return probs

    @cache
    def angle_diff(self, qubit):
        _, angle_new = misc.auto_rotate(self.df[f'cplx_{qubit}'], True)
        angle, _ = self.get_ang_thres(qubit)
        angle_diff = (angle - angle_new) % np.pi  # in [0,pi)
        return angle_diff

    def close_enouth(self, qubit, tolerance=8):
        angle_diff = self.angle_diff(qubit)
        tolerance = tolerance * np.pi/180
        close_enough = (angle_diff <= tolerance) or (np.pi - angle_diff <= tolerance)
        return close_enough

    def plot_1q(self, qubit, ax=None):
        df = self.df
        if ax is None: _, ax = plt.subplots(figsize=(3,3))

        plotter.plot_iq(df[f'cplx_{qubit}_rot'][~df[f'{qubit}_s1']], ax=ax)
        plotter.plot_iq(df[f'cplx_{qubit}_rot'][ df[f'{qubit}_s1']], ax=ax)
        _, thres = self.get_ang_thres(qubit)
        plotter.cursor(ax, x=round(thres, 3))

        # Plot the angle difference.
        angle_diff = self.angle_diff(qubit)
        x = np.linspace(*ax.get_xlim(), 5)[1:-1]
        mean = df[f'cplx_{qubit}_rot'].mean()
        y = mean.imag + np.tan(angle_diff) * (x-mean.real)
        ax.plot(x,y,'k--')
        ax.plot(x,np.ones(x.shape)*y[0], 'k-')
        ax.annotate('{:.1f} deg.'.format(angle_diff * 180/np.pi), 
                    (x[1], (y[0]+y[-1])/2), va='center')
        ax.set_title(self.lf.name.as_plot_title(qubit=qubit))

    def plot(self):
        """Plot judge for all qubits."""
        fig, axs = plt.subplots(ncols=len(self.qubits), figsize=(9,3))
        for ax, qn in zip(axs, self.qubits):
            if f'cplx_{qn}' not in self.df: continue
            self.plot_1q(qn, ax)
            ax.set(
                title=qn,
                xlabel='',
                ylabel='',
            )
            ax.tick_params(direction='in')
        fig.suptitle(self.lf.name.as_plot_title())
        

class qst_1q:
    def __init__(self, folder, id0, idx, idy, suffix='csv_complete'):
        self.ops = list('ixy')
        self.ids = [id0, idx, idy]
        self.suffix = suffix

        # Do NOT keep ss datas to save memory.
        datas = [
            single_shot_data(folder, id, suffix, c_qubits=('q1', 'q2'), p_qubits=('q5'))
            for id in self.ids
        ]
        # Fix logfiles missing |0> center or |1> center in conf.
        passed_lf = None
        for data in datas:
            if data.probs is not None:
                passed_lf = data.lf
        if passed_lf is not None:
            for data in datas:
                if data.probs is None:
                    data.lf.conf = passed_lf.conf
                    data.construct()

        self.probs = [ss_data.probs for ss_data in datas]
    
    def rho(self, select, ro_mat=None):
        probs = [(df.loc[f'q1q2_{select}', 'q5_0'], 
                  df.loc[f'q1q2_{select}', 'q5_1']) 
                 for df in self.probs]

        if ro_mat is not None:
            for i, ps in enumerate(probs):
                probs[i] = np.dot(np.linalg.inv(ro_mat), ps)
            
        rho = tomo.qst(np.array(probs), 'tomo')
        return rho


# Define constant matrices
selects = ['00', '01', '10', '11']
rho_in = {
    '0': np.array([
        [1,0],
        [0,0],
    ]),
    'x': np.array([
        [.5, .5j],
        [-.5j, .5],
    ]),
    'y': np.array([
        [.5, .5],
        [.5, .5],
    ]),
    '1': np.array([
        [0,0],
        [0,1],
    ]),
}

rho_out_ideal_ps = {
    '00': rho_in,
    # '00': {  # after I, for 00
    #     '0': rho(1,0),
    #     'x': rho(1/np.sqrt(2),-1j/np.sqrt(2)),
    #     'y': rho(1/np.sqrt(2), 1/np.sqrt(2)),
    #     '1': rho(0,1),
    # },
    '01': {  # after Ypi, for 01
        '0': rho(0,1),
        'x': rho(1/np.sqrt(2), 1j/np.sqrt(2)),
        'y': rho(1/np.sqrt(2), 1/np.sqrt(2)),
        '1': rho(1,0),
    },
    '10': {  # after YpiXpi, for 10
        '0': rho(1,0),
        'x': rho(1/np.sqrt(2), 1j/np.sqrt(2)),
        'y': rho(1/np.sqrt(2),-1/np.sqrt(2)),
        '1': rho(0,1),
    },
    '11': {  # after Xpi, for 11
        '0': rho(0,1),
        'x': rho(1/np.sqrt(2), -1j/np.sqrt(2)),
        'y': rho(1/np.sqrt(2), -1/np.sqrt(2)),
        '1': rho(1,0),
    }
}

chi_ideal_ps = {
    select: tomo.qpt(
        [rho_in[init] for init in '0xy1'], 
        [rho_out_ideal_ps[select][init] for init in '0xy1'], 
        'sigma',
    ) 
    for select in selects
}

rho_out_ideal_fb = {
    '00': rho_in,
    '01': rho_in,
    '10': rho_in,
    '11': rho_in,
}

chi_ideal_fb = {
    select: tomo.qpt(
        [rho_in[init] for init in '0xy1'], 
        [rho_out_ideal_fb[select][init] for init in '0xy1'], 
        'sigma',
    ) 
    for select in selects
}

class qpt_1q:
    """Process single shot datas in state teleportation experiments.
    
    Basically a function, but store returns in an object and provides some plot functions.

    Note:
    What the attributes looks like. Do replace & DO NOT modify them!
    ```
        .rho_in = {'0': np.array, 'x': np.array, 'y': np.array, '1': np.array}
        .qst_out = {'0': qst_2q, 'x': qst_2q, 'y': qst_2q, '1': qst_2q}

        .rho_out/.rho_out_ideal = {
            '00': {'0': np.array, 'x': np.array, 'y': np.array, '1': np.array},
            '01': {'0': np.array, 'x': np.array, 'y': np.array, '1': np.array},
            '10': {'0': np.array, 'x': np.array, 'y': np.array, '1': np.array},
            '11': {'0': np.array, 'x': np.array, 'y': np.array, '1': np.array},
        }

        .Frho = {
            '00': {'0': float, 'x': float, 'y': float, '1': float},
            '01': {'0': float, 'x': float, 'y': float, '1': float},
            '10': {'0': float, 'x': float, 'y': float, '1': float},
            '11': {'0': float, 'x': float, 'y': float, '1': float},
        }

        .chi/.chi_ideal = {
            '00': np.array,
            '01': np.array,
            '10': np.array,
            '11': np.array,
        }

        .Fchi = {
            '00': float,
            '01': float,
            '10': float,
            '11': float,
        }
    ```
    """
    def __init__(self, folder, m, suffix='csv_complete'):
        self.selects = selects

        fname = fileio.LabradRead(folder, m).name
        if 'tele_ss_ps' in fname.title.lower():
            rho_out_ideal = rho_out_ideal_ps
            chi_ideal = chi_ideal_ps
        else:
            rho_out_ideal = rho_out_ideal_fb
            chi_ideal = chi_ideal_fb
        self.rho_in = rho_in
        self.rho_out_ideal = rho_out_ideal
        self.chi_ideal = chi_ideal

        # Construct density matrices.
        qst_out = {
            '0': qst_1q(folder, m+0, m+1, m+2, suffix=suffix),
            'x': qst_1q(folder, m+3, m+4, m+5, suffix=suffix),
            'y': qst_1q(folder, m+6, m+7, m+8, suffix=suffix),
            '1': qst_1q(folder, m+9, m+10, m+11, suffix=suffix),
        }
        self.qst_out = qst_out

        rho_out = self.empty_rho_dict()
        for select, init in self.iter_rho_dict(rho_out):
            rho_out[select][init] = qst_out[init].rho(select)
        self.rho_out = rho_out

        Frho = self.empty_rho_dict()
        for select, init in self.iter_rho_dict(Frho):
            Frho[select][init] = fidelity(rho_out[select][init], 
                                          rho_out_ideal[select][init])
        self.Frho = Frho
        
        # Construct process matrices.
        chi = {
            select: tomo.qpt(
                [rho_in[init] for init in '0xy1'], 
                [rho_out[select][init] for init in '0xy1'], 
                'sigma',
            ) 
            for select in self.selects
        }
        self.chi = chi

        # Fchi = {select: fidelity(chi[select], chi_ideal[select]) 
        #         for select in selects}
        Fchi = {select: np.abs(chi[select]).max() for select in self.selects}
        self.Fchi = Fchi

        fchi_mean = np.mean(list(Fchi.values()))
        fname.title = fname.title + f', Fchi_mean={fchi_mean:.2%}'
        self.fname = fname

    def empty_rho_dict(self):
        return {select: {init: None for init in '0xy1'} for select in self.selects}
        
    @staticmethod
    def iter_rho_dict(rho_dict):
        for select, d in rho_dict.items():
            for init in d.keys():
                yield select, init

    def plot_chi(self, chi=None, title=None):
        """Plot process matrices for all selects.
        
        Note:
            - try `fig.savefig(fname.as_file_name()+'.png', bbox_inches='tight')`
        """
        if chi is None: chi = self.chi
        if title is None: title = self.fname.as_plot_title(width=100)
        lbs = list('IXYZ')

        fig = plt.figure(figsize=(15,4), tight_layout=False)
        for i, (select, mat) in enumerate(chi.items()):
            ax = fig.add_subplot(1, 4, i+1, projection='3d')
            plotter.plot_mat3d(mat, ax=ax, colorbar=False)
            ax.set_title(f'select={select}, Fchi={self.Fchi[select]:.2%}')
            ax.collections[0].set_linewidth(0.2)
            ax.set_xticklabels(lbs)
            ax.set_yticklabels(lbs)

        # Add colorbar.
        fig.subplots_adjust(left=0.1)
        cax = fig.add_axes([0.05, 0.15, 0.008, 0.6])
        cbar = fig.colorbar(ax.collections[0], cax=cax, orientation='vertical')
        cbar.set_ticks(np.linspace(-np.pi, np.pi, 5))
        cbar.set_ticklabels((r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'))
        
        fig.suptitle(title)
        return fig

    def plot_chi_4x2(self, chi=None, title=None):
        if chi is None: chi = self.chi
        if title is None: title = self.fname.as_plot_title(width=100)
        lbs = list('IXYZ')

        fig = plt.figure(figsize=(14,8), tight_layout=False)
        for i, (select, mat) in enumerate(chi.items()):
            ax_r = fig.add_subplot(2, 4, 2*i+1, projection='3d')
            ax_i = fig.add_subplot(2, 4, 2*i+2, projection='3d')
            plotter.plot_complex_mat3d(mat, [ax_r, ax_i], cmin=-1, cmax=1, colorbar=False)
            ax_r.set_title(f'select={select}, Fchi={self.Fchi[select]:.2%}')
            ax_r.collections[0].set_linewidth(0.2)
            ax_r.set_xticklabels(lbs)
            ax_r.set_yticklabels(lbs)
            ax_i.collections[0].set_linewidth(0.2)
            ax_i.set_xticklabels(lbs)
            ax_i.set_yticklabels(lbs)

        fig.subplots_adjust(top=0.9)
        cax = fig.add_axes([0.4, 0.53, 0.2, 0.01])
        cbar = fig.colorbar(ax_r.collections[0], cax=cax, orientation='horizontal')
        
        fig.suptitle(title)
        return fig


if __name__ == '__main__':
    DIR = '//XLD2-PC3/data/crab.dir/220724.dir/0919_tele.dir'

    qpt = qpt_1q(DIR, 526)
    qpt.plot_chi()
    # qpt.plot_chi_4x2()
    plt.show()
