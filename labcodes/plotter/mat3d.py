"""Functions for plotting matrice."""

from itertools import product
from typing import Callable, Literal, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
import numpy as np
from labcodes.plotter import misc

qutip_cmap = mpl.colors.LinearSegmentedColormap(
    'phase_colormap', 
    {'blue': ((0.00, 0.0, 0.0),
            (0.25, 0.0, 0.0),
            (0.50, 1.0, 1.0),
            (0.75, 1.0, 1.0),
            (1.00, 0.0, 0.0)),
    'green': ((0.00, 0.0, 0.0),
            (0.25, 1.0, 1.0),
            (0.50, 0.0, 0.0),
            (0.75, 1.0, 1.0),
            (1.00, 0.0, 0.0)),
    'red': ((0.00, 1.0, 1.0),
            (0.25, 0.5, 0.5),
            (0.50, 0.0, 0.0),
            (0.75, 0.0, 0.0),
            (1.00, 1.0, 1.0))}, 
    256,
)
try:
    import cmocean
    cmap_phase = cmocean.cm.phase
except ImportError:
    cmap_phase = None


def plot_mat(
    mat: np.ndarray,
    zmax: float = None,
    zmin: float = None,
    ax: plt.Axes = None,
    cmap='RdBu_r',
    fmt: Callable[[float], str] = '{:.2f}'.format,
    omit_below: Union[float, None] = None,
    origin: Literal['lower', 'upper'] = 'upper',
    vary_size: bool = False,
) -> plt.Axes:
    """Plot matrix values in a 2d grid.
    
    >>> plot_mat(np.random.rand(4, 4) - 0.5)
    <Axes: >
    """
    if ax is None: fig, ax = plt.subplots(figsize=(3,3))
    if zmax is None: zmax = np.max(mat)
    if zmin is None: zmin = np.min(mat)
    xdim, ydim = mat.shape
    cmap = mpl.colormaps.get_cmap(cmap)
    norm, extend_cbar = misc.get_norm(mat, cmin=zmin, cmax=zmax)
    if vary_size:
        size = np.abs(mat).clip(0, zmax) / zmax * 0.9 + 0.1
    else:
        size = np.ones_like(mat)

    squares = []
    colors = []
    for x, y in product(range(xdim), range(ydim)):
        v = mat[x, y]
        s = size[x, y]
        if omit_below is not None:
            if np.abs(v) <= omit_below: continue
        squares.append(mpl.patches.Rectangle((x-s/2, y-s/2), s, s))
        c = cmap(norm(v))
        colors.append(c)
        # Use black text if squares are light; otherwise white. See https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.convert
        txt_color = 'k' if (.299*c[0] + .587*c[1] + .114*c[2]) > 0.5 else 'w'
        txt = ax.annotate(fmt(v), (x, y), ha='center', va='center', color=txt_color)
        stroke_color = 'k' if txt_color == 'w' else 'w'
        txt.set_path_effects(
            [patheffects.withStroke(linewidth=1, foreground=stroke_color, alpha=.5)])

    col = mpl.collections.PatchCollection(squares, facecolors=colors, cmap=cmap, 
                                          norm=norm, linewidth=0)
    ax.add_collection(col)
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(1))
    ax.set_aspect('equal')
    if origin == 'upper':
        ax.tick_params(labelbottom=False, labeltop=True, direction='in')
        ax.invert_yaxis()
    ax.margins(0)
    ax.autoscale_view()
    return ax


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

class MatEditor:
    """For conveniently viewing and editing matrix.
    
    Intended for ztalk matrix manipulation.
    
    Examples:
    ```
    view = MatEditor(s['ztalk'], s['zspace'])
    view.show()

    view['Q1_by_C12'] = 0.01
    view.show()
    s['ztalk'] = view.mat
    view.close()
    ```
    """
    def __init__(self, mat, xlabels, ylabels=None):
        self.mat = np.array(mat)
        self.xlabels = list(xlabels)
        if ylabels is None:
            self.ylabels = list(xlabels).copy()
        else:
            self.ylabels = list(ylabels)
        self.fig = None
        self._interactive = plt.isinteractive()

    def __getitem__(self, key:str):
        xl, yl = key.split('_by_')
        xi = self.xlabels.index(xl)
        yi = self.ylabels.index(yl)
        return self.mat[xi,yi]
    
    def __setitem__(self, key:str, val):
        xl, yl = key.split('_by_')
        xi = self.xlabels.index(xl)
        yi = self.ylabels.index(yl)
        self.mat[xi,yi] = val

    def show(self, omit_diag=True, figsize_scale=0.8):
        """Show the matrix in a figure. Diagonal terms are omitted by default."""
        vals = self.mat.copy()
        xdim, ydim = vals.shape
        xax = np.arange(xdim)
        yax = np.arange(ydim)
        xgrid, ygrid = np.meshgrid(xax, yax)
        xlabels, ylabels = self.xlabels, self.ylabels

        if omit_diag is True:
            for i in range(min(xdim, ydim)):
                vals[i,i] = 0
        mask = (np.abs(vals) >= 0.2)
        vmax = np.max(np.abs(vals[~mask]))

        if self.fig is None:
            fig, ax = plt.subplots(figsize=(xdim*figsize_scale, ydim*figsize_scale))
            self.fig = fig
            self._interactive = plt.isinteractive()
            plt.ion()
        else:
            fig = self.fig
            ax = fig.gca()
            ax.clear()
        ax.grid(lw=1, alpha=0.5)
        ax.set_xticks(xax)
        ax.set_yticks(yax)
        ax.set_xlim(-0.5, xdim - 0.5)
        ax.set_ylim(-0.5, ydim - 0.5)
        ax.set_xticklabels(xlabels)
        ax.set_yticklabels(ylabels)
        ax.tick_params(labelbottom=False, labeltop=True, direction='in')
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.scatter(xgrid[~mask], ygrid[~mask], np.abs(vals[~mask])*3e4, vals[~mask], 
                   cmap='bwr', vmin=-vmax, vmax=vmax, marker='s')
        ax.scatter(xgrid[mask], ygrid[mask], 1500, 'k', marker='s')
        for x, y, v in zip(xgrid.ravel(), ygrid.ravel(), vals.ravel()):
            if v == 0: continue
            color = 'k' if abs(v) <= 0.2 else 'w'
            ax.annotate(f'{v:.1%}\n{ylabels[y]} by {xlabels[x]}', (x,y), 
                        size='small', ha='center', va='center', color=color)
        return ax
    
    def close(self):
        """Close the figure and restore the interactive state."""
        if self.fig is None: return
        plt.close(self.fig)
        self.fig = None
        if self._interactive is False:
            plt.ioff()
        elif self._interactive is True:
            plt.ion()

def _plot_mat3d(ax, mat, cval, cmin=None, cmax=None, cmap='bwr', alpha=1.0, label=True, fmt=None):
    if fmt is None: fmt = lambda v: f'{v:.3f}'.replace('0.', '.')
    if cmap == 'qutip': cmap = qutip_cmap

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
        mat = mat.real
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
        if cmap is None: cmap = cmap_phase or 'twilight'
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


if __name__ == "__main__":
    import doctest
    doctest.testmod()