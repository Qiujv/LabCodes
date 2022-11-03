"""Process single shot tomo datas, for state teleport and tomo teleport experiments."""

from functools import cache
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from labcodes import fileio, misc, plotter
import labcodes.frog.pyle_tomo as tomo


class single_shot_data:
    def __init__(self, folder, id, suffix='csv', c_qubits=('q1', 'q2'), p_qubits=('q5',)):
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

    def construct(self):
        df = self.lf.df.copy()
        for qn in self.qubits:
            if f'{qn}_s1' in df:
                df[f'{qn}_s1'] = df[f'{qn}_s1'].astype(bool)
            else:
                df = self.judge(df, qn, drop=True)
        self.df = df
        self.probs = self.get_probs()

    def get_center(self, qubit, state):
        """Get |0> center or |1> center fron logf.conf, return in a complex number."""
        center = self.lf.conf['parameter'][f'Device.{qubit.upper()}.|{state}> center']['data'][20:-2].split(', ')
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
        df: columns[p0, p1], index=[c00, c01, c10, c11].
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
        if ax is None: _, ax = plt.subplots()

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

class qpt_1q:
    """Process single shot datas in state teleportation experiments.
    
    Basically a function, but store returns in an object and provides some plot functions.
    """
    def __init__(self, folder, m, suffix='csv_complete'):
        selects = ['00', '01', '10', '11']
        self.selects = selects

        # Construct density matrices.
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
                [.5, .5]
            ]),
            '1': np.array([
                [0,0],
                [0,1],
            ]),
        }
        self.rho_in = rho_in

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

        rho_out_ideal = {select: rho_in for select in selects}
        # rho_out_ideal = {
        #     '00': rho_in,
        #     # '00': {  # after I, for 00
        #     #     '0': rho(1,0),
        #     #     'x': rho(1/np.sqrt(2),-1j/np.sqrt(2)),
        #     #     'y': rho(1/np.sqrt(2), 1/np.sqrt(2)),
        #     #     '1': rho(0,1),
        #     # },
        #     '01': {  # after Ypi, for 01
        #         '0': rho(0,1),
        #         'x': rho(1/np.sqrt(2), 1j/np.sqrt(2)),
        #         'y': rho(1/np.sqrt(2), 1/np.sqrt(2)),
        #         '1': rho(1,0),
        #     },
        #     '10': {  # after YpiXpi, for 10
        #         '0': rho(1,0),
        #         'x': rho(1/np.sqrt(2), 1j/np.sqrt(2)),
        #         'y': rho(1/np.sqrt(2),-1/np.sqrt(2)),
        #         '1': rho(0,1),
        #     },
        #     '11': {  # after Xpi, for 11
        #         '0': rho(0,1),
        #         'x': rho(1/np.sqrt(2), -1j/np.sqrt(2)),
        #         'y': rho(1/np.sqrt(2), -1/np.sqrt(2)),
        #         '1': rho(1,0),
        #     }
        # }
        self.rho_out_ideal = rho_out_ideal

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
            for select in selects
        }
        self.chi = chi

        chi_ideal = {
            select: tomo.qpt(
                [rho_in[init] for init in '0xy1'], 
                [rho_out_ideal[select][init] for init in '0xy1'], 
                'sigma',
            ) 
            for select in selects
        }
        self.chi_ideal = chi_ideal

        # Fchi = {select: fidelity(chi[select], chi_ideal[select]) 
        #         for select in selects}
        Fchi = {select: np.abs(chi[select]).max() for select in selects}
        self.Fchi = Fchi

    def empty_rho_dict(self):
        return {select: {init: None for init in '0xy1'} for select in self.selects}
        
    @staticmethod
    def iter_rho_dict(rho_dict):
        for select, d in rho_dict.items():
            for init in d.keys():
                yield select, init

    def plot_chi(self):
        pass

def rho(alpha, beta):
    """Returns density matrix of qubit state alpha*|0> + beta*|1>"""
    state = alpha*np.array([[1],[0]]) + beta*np.array([[0],[1]])
    state = np.matrix(state)
    return np.array(np.dot(state, state.H))


def fidelity(rho, sigma):
    return np.real(np.trace(np.dot(rho, sigma)))