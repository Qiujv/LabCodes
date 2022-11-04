"""Process single shot tomo datas, for state teleport experiments."""

from functools import cache
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from labcodes import fileio, misc, plotter
import labcodes.frog.pyle_tomo as tomo
from joblib import Parallel, delayed
from labcodes.frog import state_list
from labcodes.frog.tele_state import single_shot_data, fidelity, rho


class qst_2q:
    def __init__(self, folder, m, suffix='csv_complete'):
        # self.ops = list('ixy')
        # self.ids = [id0, idx, idy]
        self.suffix = suffix

        # Do NOT keep ss datas to save memory.
        datas = [
            single_shot_data(folder, id, suffix, c_qubits=['q2', 'q5'], p_qubits=['q1', 'q4'])
            for id in range(m, m+9)
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
        probs = [(df.loc[f'q2q5_{select}', 'q1q4_00'], 
                  df.loc[f'q2q5_{select}', 'q1q4_01'], 
                  df.loc[f'q2q5_{select}', 'q1q4_10'], 
                  df.loc[f'q2q5_{select}', 'q1q4_11']) 
                 for df in self.probs]

        if ro_mat is not None:
            for i, ps in enumerate(probs):
                probs[i] = np.dot(np.linalg.inv(ro_mat), ps)
            
        rho = tomo.qst(np.array(probs), 'tomo2')
        return rho

selects = ['00', '01', '10', '11']
init_states = [''.join(init) for init in product('0xy1', repeat=2)]

rho_in = dict(zip(init_states, state_list.rho_in))

rho_out_ideal_ps = {
    select: dict(zip(init_states, state_list.rho_out_ideal[select]))
    for select in selects
}

chi_ideal_ps = {
    select: tomo.qpt(
        [rho_in[init] for init in init_states], 
        [rho_out_ideal_ps[select][init] for init in init_states], 
        'sigma2',
    ) 
    for select in selects
}

rho_out_ideal_fb = {
    select: dict(zip(init_states, state_list.rho_out_ideal_fb[select]))
    for select in selects
}

chi_ideal_fb = {
    select: tomo.qpt(
        [rho_in[init] for init in init_states], 
        [rho_out_ideal_fb[select][init] for init in init_states], 
        'sigma2',
    ) 
    for select in selects
}

class qpt_2q:
    """Process single shot datas in gate teleportation experiments.
    
    Basically a function, but store returns in an object and provides some plot functions.
    """
    def __init__(self, folder, m, suffix='csv_complete', parallel=True):
        fname = fileio.LabradRead(folder, m).name
        self.selects = selects
        self.init_states = init_states

        # Construct density matrices.
        qst_out = {  # Slow, but cannot merge with jobs below.
            '00': qst_2q(folder, m+0, suffix=suffix),
            '0x': qst_2q(folder, m+9, suffix=suffix),
            '0y': qst_2q(folder, m+18, suffix=suffix),
            '01': qst_2q(folder, m+27, suffix=suffix),

            'x0': qst_2q(folder, m+36, suffix=suffix),
            'xx': qst_2q(folder, m+45, suffix=suffix),
            'xy': qst_2q(folder, m+54, suffix=suffix),
            'x1': qst_2q(folder, m+63, suffix=suffix),

            'y0': qst_2q(folder, m+72, suffix=suffix),
            'yx': qst_2q(folder, m+81, suffix=suffix),
            'yy': qst_2q(folder, m+90, suffix=suffix),
            'y1': qst_2q(folder, m+99, suffix=suffix),

            '10': qst_2q(folder, m+108, suffix=suffix),
            '1x': qst_2q(folder, m+117, suffix=suffix),
            '1y': qst_2q(folder, m+126, suffix=suffix),
            '11': qst_2q(folder, m+135, suffix=suffix),
        }
        self.qst_out = qst_out

        rho_out = self.empty_rho_dict()
        if parallel:
            def job(select, init):
                return select, init, qst_out[init].rho(select)
            res = Parallel(n_jobs=8, verbose=10)(
                delayed(job)(select, init)
                for select, init in self.iter_rho_dict(rho_out)
            )
            for select, init, rho_one in res:
                rho_out[select][init] = rho_one
        else:
            for select, init in self.iter_rho_dict(rho_out):
                rho_out[select][init] = qst_out[init].rho(select)
        self.rho_out = rho_out
        

        if '_FB' in fname.title.lower():  # TODO: fix this.
            rho_out_ideal = rho_out_ideal_fb
            chi_ideal = chi_ideal_fb
        else:
            rho_out_ideal = rho_out_ideal_ps
            chi_ideal = chi_ideal_ps
        self.rho_in = rho_in
        self.rho_out_ideal = rho_out_ideal
        self.chi_ideal = chi_ideal

        Frho = self.empty_rho_dict()
        for select, init in self.iter_rho_dict(Frho):
            Frho[select][init] = fidelity(rho_out[select][init], 
                                          rho_out_ideal[select][init])
        self.Frho = Frho
        
        # Construct process matrices.
        chi = {
            select: tomo.qpt(
                [rho_in[init] for init in init_states], 
                [rho_out[select][init] for init in init_states], 
                'sigma2',
            ) 
            for select in selects
        }
        self.chi = chi

        Fchi = {select: fidelity(chi[select], chi_ideal[select]) 
                for select in selects}
        self.Fchi = Fchi

        fchi_mean = np.mean(list(Fchi.values()))
        fname.title = fname.title + f', Fchi_mean={fchi_mean:.2%}'
        self.fname = fname

    def empty_rho_dict(self):
        return {select: {init: None for init in self.init_states} for select in self.selects}
        
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
        lbs = [''.join([op1, op2]) for op1, op2 in product('IXYZ', repeat=2)]

        fig = plt.figure(figsize=(10,10), tight_layout=False)
        for i, (select, mat) in enumerate(chi.items()):
            ax = fig.add_subplot(2, 2, i+1, projection='3d')
            plotter.plot_mat3d(mat, ax=ax, colorbar=False, label=False, cmap='qutip')
            ax.set_title(f'select={select}, Fchi={self.Fchi[select]:.2%}')
            ax.collections[0].set_linewidth(0.2)
            ax.set_zlim(0,0.25)
            ax.tick_params('both', pad=0, labelsize='small')
            ax.tick_params('x', labelrotation=45)
            ax.tick_params('y', labelrotation=-45)
            ax.set_xticklabels(lbs)
            ax.set_yticklabels(lbs)
            ax.set_zticks([0, 0.25])

        # Add colorbar.
        cax = fig.add_axes([0.4, 0.53, 0.2, 0.01])
        cbar = fig.colorbar(ax.collections[0], cax=cax, orientation='horizontal')
        cbar.set_ticks(np.linspace(-np.pi, np.pi, 5))
        cbar.set_ticklabels((r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'))
        
        fig.suptitle(title)
        return fig

    def plot_chi_4x2(self, chi=None, title=None):
        if chi is None: chi = self.chi
        if title is None: title = self.fname.as_plot_title(width=100)
        lbs = [''.join([op1, op2]) for op1, op2 in product('IXYZ', repeat=2)]

        fig = plt.figure(figsize=(20,10), tight_layout=False)
        for i, (select, mat) in enumerate(chi.items()):
            ax_r = fig.add_subplot(2, 4, 2*i+1, projection='3d')
            ax_i = fig.add_subplot(2, 4, 2*i+2, projection='3d')
            plotter.plot_complex_mat3d(mat, [ax_r, ax_i], cmin=-0.25, cmax=0.25, 
                                    colorbar=False, label=False)
            ax_r.set_title(f'select={select}, Fchi={self.Fchi[select]:.2%}')
            for ax in ax_r, ax_i:
                ax.set_zlim(-0.25,0.25)
                ax.set_zticks([0, 0.25])
                ax.collections[0].set_linewidth(0.2)
                ax.set_xticklabels(lbs)
                ax.set_yticklabels(lbs)
                ax.tick_params('both', pad=0, labelsize='small')
                ax.tick_params('x', labelrotation=45)
                ax.tick_params('y', labelrotation=-45)

        fig.subplots_adjust(top=0.9)
        cax = fig.add_axes([0.4, 0.53, 0.2, 0.01])
        cbar = fig.colorbar(ax_r.collections[0], cax=cax, orientation='horizontal')
        
        fig.suptitle(title)
        return fig

if __name__ == '__main__':
    DIR = '//XLD2-PC3/data/crab.dir/220724_2.dir/1024_gate_fb.dir'

    qpt = qpt_2q(DIR, 8397)
    qpt.plot_chi()
    # qpt.plot_chi_4x2()
    plt.show()
