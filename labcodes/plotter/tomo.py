"""Functions for plotting datas by tomo experiment."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def plot_density_matrix_real(rho, ax=None, cmap='jet', alpha=0.6):
    """Plot 3d bar for real(density matrix)."""
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1,projection='3d')
        ax.view_init(azim=-28, elev=65)
    else:
        fig = ax.get_figure()

    bar_width = 0.6
    xpos, ypos = np.meshgrid(
        np.arange(1, rho.shape[0] + 1, 1),
        np.arange(1, rho.shape[1] + 1, 1)
    )
    xpos = xpos.T.flatten() - bar_width/2
    ypos = ypos.T.flatten() - bar_width/2
    zpos = np.zeros(rho.size)
    dx = dy = bar_width * np.ones(rho.size)
    dz = rho.real.flatten()

    norm = mpl.colors.Normalize(-1, 1)
    cmap = plt.get_cmap(cmap)
    colors = cmap(norm(dz))

    bar_col = ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, alpha=alpha, 
                       cmap=cmap, norm=norm, edgecolor='white', linewidth=1)

    ax.axes.w_xaxis.set_major_locator(plt.IndexLocator(1, bar_width/2))
    ax.axes.w_yaxis.set_major_locator(plt.IndexLocator(1, bar_width/2))
    ax.axes.w_zaxis.set_major_locator(plt.IndexLocator(1, 0.5))
    ax.set(
        zlim3d=(-0.5, 0.5),
        zticks=np.arange(-0.5, 1.5, 0.5),
    )

    cbar = fig.colorbar(bar_col, ax=ax, shrink=0.6, pad=0.1)
    return ax

def plot_density_matrix(rho, ticklabels=None, cmap='jet', alpha=0.6):
    """Plot 3d bar for density matrix, both the real and imag part.
    
    Create a new figure and returns ax_real, ax_imag.
    """
    fig = plt.figure(figsize=(9,4))
    ax_real = fig.add_subplot(1,2,1,projection='3d')
    plot_density_matrix_real(rho.real, ax=ax_real)
    ax_real.collections[-1].colorbar.remove()
    ax_imag = fig.add_subplot(1,2,2,projection='3d')
    plot_density_matrix_real(rho.imag, ax=ax_imag)
    if ticklabels:
        ax_real.set(
            xticklabels=ticklabels,
            yticklabels=ticklabels,
        )
        ax_imag.set(
            xticklabels=ticklabels,
            yticklabels=ticklabels,
        )
    return ax_real, ax_imag
