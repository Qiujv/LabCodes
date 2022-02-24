"""Functions for plotting datas by tomo experiment."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def plot_mat3d(mat, ax=None, view_angle=(None, None), cmap='bwr', alpha=1.0, 
    cmin=None, cmax=None, colorbar=True):
    """Plot 3d bar for matrix. 
    
    Args:
        view_angle, passed to ax.view_init.
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
    dz = mat.real.flatten()

    if isinstance(cmap, str):
        cmap = plt.cm.get_cmap(cmap)
    adjust_clims = True if (cmin is None) or (cmax is None) else False
    zmin, zmax = dz.min(), dz.max()
    if cmin is None:
        cmin = zmin
    if cmax is None:
        cmax = zmax
    if adjust_clims:
        cmax = max(abs(cmin), abs(cmax)); cmin = -cmax  # Make sure 0 in the mid.
    if (zmin < cmin) and (zmax > cmax):
        extend_cbar = 'both'
    elif zmin < cmin:
        extend_cbar = 'min'
    elif zmax > cmax:
        extend_cbar = 'max'
    else:
        extend_cbar = 'neither'
    norm = mpl.colors.Normalize(cmin, cmax)
    colors = cmap(norm(dz))

    bar_col = ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, alpha=alpha, 
                       cmap=cmap, norm=norm, edgecolor='white', linewidth=1)

    ax.set(
        xticks=np.arange(1, mat.shape[0] + 1, 1),
        yticks=np.arange(1, mat.shape[1] + 1, 1),
        zticks=np.arange(0.5*(zmax//0.5 + 1), 0.5*(zmin//0.5 - 1), -0.5),
    )

    if colorbar is True:
        # Way to remove colorbar: ax.collections[-1].colorbar.remove()
        cbar = fig.colorbar(bar_col, shrink=0.6, pad=0.1, extend=extend_cbar)
    return ax

def 

def plot_complex_mat3d(mat, axs=None, **kwargs):
    """Plot 3d bar for complex matrix, both the real and imag part.
    """
    if axs is None:
        fig = plt.figure(figsize=(9,4))
        ax_real = fig.add_subplot(1,2,1,projection='3d')
        ax_imag = fig.add_subplot(1,2,2,projection='3d')
    else:
        ax_real, ax_imag = axs
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.95, 0.15, 0.05, 0.7])

    cmin = 
    plot_mat3d(mat.real, ax=ax_real, colorbar=False, **kwargs)
    plot_mat3d(mat.imag, ax=ax_imag, colorbar=False, **kwargs)

    return ax_real, ax_imag
