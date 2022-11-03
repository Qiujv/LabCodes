"""Functions for plotting matrice."""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from labcodes.plotter import misc
from matplotlib.ticker import EngFormatter


def plot_mat2d(mat, txt=None, fmt='{:.2f}'.format, ax=None, cmap='binary', **kwargs):
    """Plot matrix values in a 2d grid."""
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
        
    mat = np.array(mat)
    if txt is None:
        txt = [fmt(i) for i in mat.ravel()]
        txt = np.array(txt).reshape(mat.shape)

    ax.matshow(mat, cmap=cmap, **kwargs)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.annotate(txt[i,j], (j, i), ha='center', va='center', backgroundcolor='w')

    return ax

def _plot_mat3d(ax, mat, cval, cmin=None, cmax=None, cmap='bwr', alpha=1.0, label=True, fmt=None):
    if fmt is None: fmt = lambda v: f'{v:.3f}'.replace('0.', '.')

    bar_width = 0.6
    xpos, ypos = np.meshgrid(
        np.arange(1, mat.shape[0] + 1, 1),
        np.arange(1, mat.shape[1] + 1, 1)
    )
    xpos = xpos.T.flatten() - bar_width/2
    ypos = ypos.T.flatten() - bar_width/2
    zpos = np.zeros(mat.size)
    dx = dy = bar_width * np.ones(mat.size)
    dz = mat.flatten()

    symmetric_clims = (cmin is None) or (cmax is None)
    norm, extend_cbar = misc.get_norm(cval.flatten(), cmin=cmin, cmax=cmax, symmetric=symmetric_clims)
    cmap = plt.cm.get_cmap(cmap)
    colors = cmap(norm(cval.flatten()))

    if alpha != 0.0:
        # Plot filled bar.
        bar_col = ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, alpha=alpha, 
                           cmap=cmap, norm=norm, edgecolor='white', linewidth=1)
    else:
        # Plot frames only.
        bar_col = ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=(0,0,0,0), alpha=None, 
                           edgecolor='black', linewidth=0.5)

    if label is True:
        for x, y, z in zip(xpos, ypos, dz):
            msg = fmt(z)
            ax.text(x+bar_width/2, y+bar_width/2, z, msg, ha='center', va='bottom',
                    bbox=dict(fill=True, color='white', alpha=0.4))

    ax.set(
        xticks=np.arange(1, mat.shape[0] + 1, 1),
        yticks=np.arange(1, mat.shape[1] + 1, 1),
        zticks=np.arange(0.5*(dz.max()//0.5 + 1), 0.5*(dz.min()//0.5 - 1), -0.5),
    )

    if np.all(dz < 0.6) and np.any(dz >= 0.5):
        ax.set_zlim(top=0.5)

    return bar_col, extend_cbar

def plot_mat3d(mat, ax=None, alpha=1.0, label=True, fmt=None,
               cmap=None, cmin=None, cmax=None, colorbar=True, cbar_location='left'):
    """Plot 3d bar for matrix.

    if alpha=0, plot bar frames only.
    
    Note:
    - To remove colorbar: `ax.collections[-1].colorbar.remove()`
    - To adjust view angle: `ax.view_init(azim=30, elev=60)`
    """
    if ax is None:
        fig = plt.figure(tight_layout=False)
        ax = fig.add_subplot(projection='3d')
        
    else:
        fig = ax.get_figure()

    if np.all(np.imag(mat) == 0):
        # Plot real matrix.
        if cmap is None: cmap = 'bwr'
        bar_col, extend_cbar = _plot_mat3d(ax, mat, cval=mat, cmin=cmin, cmax=cmax, 
                                        cmap=cmap, alpha=alpha, label=label, fmt=fmt)

        if colorbar is True:
            cbar = fig.colorbar(bar_col, shrink=0.6, fraction=0.1, pad=0.05,
                                extend=extend_cbar, location=cbar_location)
        else:
            cbar = None
    else:
        # Plot complex matrix.
        if cmap is None: cmap = 'twilight'
        if cmin is None: cmin = -np.pi
        if cmax is None: cmax = np.pi
        bar_col, extend_cbar = _plot_mat3d(ax, np.abs(mat), cval=np.angle(mat), 
                                        cmin=cmin, cmax=cmax, 
                                        cmap=cmap, alpha=alpha, label=label, fmt=fmt)

        if colorbar is True:
            cbar = fig.colorbar(bar_col, shrink=0.6, fraction=0.1, pad=0.05,
                                extend=extend_cbar, location=cbar_location)
            cbar.set_ticks(np.linspace(-np.pi, np.pi, 5))
            cbar.set_ticklabels(
                (r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'))
            # cbar.set_label('arg')
        else:
            cbar = None
    return ax

def plot_complex_mat3d(mat, axs=None, cmin=None, cmax=None, cmap='bwr', colorbar=True, **kwargs):
    """Plot 3d bar for complex matrix, both the real and imag part on two axes.
    """
    if axs is None:
        fig = plt.figure(figsize=(9,4), tight_layout=False)
        ax_real = fig.add_subplot(1,2,1,projection='3d')
        ax_imag = fig.add_subplot(1,2,2,projection='3d')
    else:
        ax_real, ax_imag = axs
        fig = ax_real.get_figure()

    norm, extend_cbar = misc.get_norm(np.hstack((mat.imag, mat.real)), cmin=cmin, cmax=cmax)
    if colorbar is True:
        fig.subplots_adjust(right=0.9)
        cax = fig.add_axes([0.95, 0.15, 0.01, 0.6])
        cmap = plt.cm.get_cmap(cmap)
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, extend=extend_cbar)
    
    kwargs.update(dict(
        cmap=cmap,
        cmin=norm.vmin,
        cmax=norm.vmax,
        colorbar=False,
    ))
    plot_mat3d(mat.real, ax=ax_real, **kwargs)
    plot_mat3d(mat.imag, ax=ax_imag, **kwargs)

    return ax_real, ax_imag
