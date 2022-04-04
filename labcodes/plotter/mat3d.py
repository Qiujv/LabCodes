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

def plot_mat3d(mat, ax=None, view_angle=(None, None), cmap='bwr', alpha=1.0, 
    cmin=None, cmax=None, colorbar=True):
    """Plot 3d bar for matrix.
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1,projection='3d')
        ax.view_init(azim=view_angle[0], elev=view_angle[1])
    else:
        fig = ax.get_figure()

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

    adjust_clims = (cmin is None) or (cmax is None)
    norm, extend_cbar = misc.get_norm(dz, cmin=cmin, cmax=cmax, symmetric=adjust_clims)
    cmap = plt.cm.get_cmap(cmap)
    colors = cmap(norm(dz))

    bar_col = ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, alpha=alpha, 
                       cmap=cmap, norm=norm, edgecolor='white', linewidth=1)

    ax.set(
        xticks=np.arange(1, mat.shape[0] + 1, 1),
        yticks=np.arange(1, mat.shape[1] + 1, 1),
        zticks=np.arange(0.5*(dz.max()//0.5 + 1), 0.5*(dz.min()//0.5 - 1), -0.5),
    )

    if colorbar is True:
        # Way to remove colorbar: ax.collections[-1].colorbar.remove()
        cbar = fig.colorbar(bar_col, shrink=0.6, pad=0.1, extend=extend_cbar)
    else:
        cbar = None
    return ax

def plot_complex_mat3d(mat, axs=None, cmin=None, cmax=None, cmap='bwr', colorbar=True, **kwargs):
    """Plot 3d bar for complex matrix, both the real and imag part.
    """
    if axs is None:
        fig = plt.figure(figsize=(9,4))
        ax_real = fig.add_subplot(1,2,1,projection='3d')
        ax_imag = fig.add_subplot(1,2,2,projection='3d')
    else:
        ax_real, ax_imag = axs

    if colorbar is True:
        fig.subplots_adjust(right=0.9)
        cax = fig.add_axes([0.95, 0.15, 0.01, 0.6])
        norm, extend_cbar = misc.get_norm(np.hstack((mat.imag, mat.real)), cmin=cmin, cmax=cmax)
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
