"""Functions for plotting matrices."""

import math
import warnings
from itertools import product
from typing import Callable, Literal, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
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
    warnings.warn("cmocean not found. Using twilight instead.", ImportWarning)
    cmap_phase = None


def plot_mat(
    mat: np.ndarray,
    zmax: float = None,
    zmin: float = None,
    ax: plt.Axes = None,
    fmt: Callable[[float], str] = '{:.2f}'.format,
    omit_below: Union[float, None] = None,
    origin: Literal['lower', 'upper'] = 'upper',
    cmap='RdBu_r',
    vary_size: bool = False,
) -> plt.Axes:
    """Plot matrix values in a 2d grid.

    Similar to plt.imshow, but with text labels.
    
    >>> plot_mat([[.9,.05],[-.1,.3]], zmax=.5, omit_below=.06, vary_size=True)
    <Axes: >
    """
    mat = np.asarray(mat).T  # Make it same as plt.imshow.
    if ax is None: fig, ax = plt.subplots(figsize=(3,3))
    if zmax is None: zmax = np.nanmax(mat)
    if zmin is None: zmin = np.nanmin(mat)
    xdim, ydim = mat.shape
    cmap = mpl.colormaps.get_cmap(cmap)
    norm, _ = misc.get_norm(mat, cmin=zmin, cmax=zmax)
    if vary_size:
        absmax = np.nanmax(np.abs(mat))
        size = np.abs(mat).clip(0, absmax) / absmax * 0.9 + 0.1
    else:
        size = np.ones_like(mat)

    squares = []
    colors = []
    for x, y in product(range(xdim), range(ydim)):
        v = mat[x, y]
        s = size[x, y]
        if np.isnan(v): continue
        if omit_below is not None:
            if np.abs(v) <= omit_below: continue
        squares.append(mpl.patches.Rectangle((x-s/2, y-s/2), s, s))
        c = cmap(norm(v))  # RGBA values.
        colors.append(c)
        # Use black text if squares are light; otherwise white.
        txt_color = 'k' if mpl.colors.rgb_to_hsv(c[:3])[2] > 0.5 else 'w'
        txt = ax.annotate(fmt(v), (x, y), ha='center', va='center', color=txt_color)
        stroke_color = 'k' if txt_color == 'w' else 'w'
        txt.set_path_effects([misc.txt_effect(color=stroke_color)])

    col = mpl.collections.PatchCollection(squares, facecolors=colors, cmap=cmap, 
                                          norm=norm, linewidth=0)
    ax.add_collection(col)
    ax.set_xlim(-0.5, xdim - 0.5)
    ax.set_ylim(-0.5, ydim - 0.5)
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(1))
    ax.set_aspect('equal')
    if origin == 'upper':
        ax.xaxis.set_label_position('top')
        ax.xaxis.set_ticks_position('top')
        ax.yaxis.set_inverted(True)
    return ax


def plot_mat2d(mat, txt=None, fmt='{:.2f}'.format, ax=None, cmap='binary', **kwargs):
    warnings.warn("plot_mat2d is deprecated. Use plot_mat instead.", DeprecationWarning)
    return plot_mat(mat, ax=ax, fmt=fmt, cmap=cmap)


def _plot_mat3d(
    ax: plt.Axes,
    mat: np.matrix,
    cval: np.ndarray,
    cmin: float = None,
    cmax: float = None,
    cmap: str = 'bwr',
    ztick_step: float = 0.5,
    alpha: float = 1,
    label: bool = True,
    fmt: Callable[[float], str] = lambda v: f'{v:.3f}'.replace('0.', '.'),
    omit_below: Union[float, None] = None,
):
    """
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(projection='3d')
    >>> mat = [[.49, .05],[-.5, .51]]
    >>> _plot_mat3d(ax, mat, mat, omit_below=.06)  # doctest: +ELLIPSIS
    (<...
    """
    mat = np.asarray(mat)
    cval = np.asarray(cval)
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
    norm, extend_cbar = misc.get_norm(cval.flatten(), cmin=cmin, cmax=cmax, 
                                      symmetric=symmetric_clims)
    cmap = mpl.colormaps.get_cmap(cmap)
    colors = cmap(norm(cval.flatten()))

    if alpha != 0:
        # Plot filled bar.
        bar_col = ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, alpha=alpha, 
                           cmap=cmap, norm=norm, edgecolor='white', linewidth=1)
    else:
        # Plot frames only.
        bar_col = ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=(0,0,0,0), alpha=None, 
                           edgecolor='black', linewidth=0.5)

    if label:
        for x, y, z in zip(xpos, ypos, dz):
            if omit_below is not None:
                if np.abs(z) <= omit_below: continue
            ax.text(x + bar_width / 2, y + bar_width / 2, z, fmt(z), 
                    ha='center', va='bottom', path_effects=[misc.txt_effect()])

    ax.set_xticks(np.arange(1, mat.shape[0] + 1, 1))
    ax.set_yticks(np.arange(1, mat.shape[1] + 1, 1))
    ax.set_zticks(np.arange(
        math.floor(dz.min() / ztick_step) * ztick_step,
        math.ceil(dz.max() / ztick_step) * ztick_step,
        ztick_step,
    ))

    return bar_col, extend_cbar

def plot_mat3d(
    mat: np.matrix,
    ax: plt.Axes = None,
    label: bool = True,
    fmt: Callable[[float], str] = lambda v: f'{v:.3f}'.replace('0.', '.'),
    omit_below: Union[float, None] = None,
    cmap: str = None,
    colorbar: bool = True,
):
    """Plot 3d bar chart for matrix.

    Example:
    >>> plot_mat3d([[.49, -.5j],[.05, .51]], omit_below=.06)
    <Axes3D: >

    Note:
    - To remove colorbar: `ax.collections[-1].colorbar.remove()`
    - To adjust view angle: `ax.view_init(azim=30, elev=60)`
    """
    mat = np.asarray(mat)
    if ax is None:
        fig = plt.figure(layout='none')
        ax = fig.add_subplot(projection='3d')
    else:
        fig = ax.get_figure()

    if np.allclose(mat.imag, 0):
        if cmap is None: cmap = 'bwr'
        _plot_mat3d(ax, mat.real, cval=mat.real, omit_below=omit_below, cmap=cmap, 
                    label=label, fmt=fmt)
    else:
        # Plot complex matrix, with color mapping the phase.
        if cmap is None: cmap = cmap_phase or 'twilight'
        col, extend = _plot_mat3d(
            ax, np.abs(mat), cval=np.angle(mat), omit_below=omit_below, cmap=cmap,
            label=label, fmt=fmt, cmin=-np.pi, cmax=np.pi,
        )
        if colorbar:
            cbar = fig.colorbar(col, shrink=0.6, fraction=0.1, pad=0.05, extend=extend)
            cbar.set_ticks(np.linspace(-np.pi, np.pi, 5))
            cbar.set_ticklabels((r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'))
    return ax


def plot_complex_mat3d(
    mat: np.matrix,
    axs: tuple[plt.Axes] = None,
    cmin: float = None,
    cmax: float = None,
    cmap: str = 'bwr',
    ztick_step: float = 0.5,
    label: bool = True,
    fmt: Callable[[float], str] = lambda v: f'{v:.3f}'.replace('0.', '.'),
    omit_below: Union[float, None] = None,
):
    """Plot 3d bar for complex matrix, both the real and imag part on two axes.
    
    >>> plot_complex_mat3d([[.49, -.5j],[.05, .51]], omit_below=.06)
    (<Axes3D: >, <Axes3D: >)
    """
    mat = np.asarray(mat)
    norm, _ = misc.get_norm(np.hstack((mat.imag, mat.real)), cmin=cmin, cmax=cmax)
    kws = dict(
        cmap=cmap,
        cmin=norm.vmin,
        cmax=norm.vmax,
        ztick_step=ztick_step,
        label=label,
        fmt=fmt,
        omit_below=omit_below,
    )

    if axs is None:
        fig = plt.figure(figsize=(9,4), layout='none')
        ax_real = fig.add_subplot(1,2,1,projection='3d')
        ax_imag = fig.add_subplot(1,2,2,projection='3d')
    else:
        ax_real, ax_imag = axs
        fig = ax_real.get_figure()
    _plot_mat3d(ax_real, mat.real, mat.real, **kws)
    _plot_mat3d(ax_imag, mat.imag, mat.imag, **kws)
    return ax_real, ax_imag


if __name__ == "__main__":
    import doctest
    doctest.testmod()
    plt.show()
