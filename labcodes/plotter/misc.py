"""Functions not fitting elsewhere."""

import matplotlib as mpl
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np


def txt_effect(color='w', linewidth=1, alpha=0.5) -> patheffects.withStroke:
    """Text effect for better visibility.
    
    Usage:
    >>> plt.text(0, 0, 'Hello world', color='w', path_effects=[txt_effect('k')])
    Text(0, 0, 'Hello world')
    """
    return patheffects.withStroke(linewidth=linewidth, foreground=color, alpha=alpha)

def plot_iq(
    data:np.ndarray,
    ax: plt.Axes = None,
    n_pt_max: int = 6000, 
    **kw_to_scatter_or_hist2d
) -> plt.Axes:
    """Plot data on complex plane. 
    Scatter when points is few, otherwise hist2d."""
    if ax is None:
        fig, ax = plt.subplots(tight_layout=True)
    else:
        fig = ax.get_figure()

    data = np.asarray(data).ravel()

    if data.size <= n_pt_max:
        kw = dict(marker='.', alpha=0.3, linewidth=0)
        kw.update(kw_to_scatter_or_hist2d)
        col = ax.scatter(np.real(data), np.imag(data), **kw)
        if data.size >= 100: col.set_rasterized(True)
    else:
        kw = dict(bins=100, norm=mpl.colors.PowerNorm(0.5))
        kw.update(kw_to_scatter_or_hist2d)
        ax.hist2d(np.real(data), np.imag(data), **kw)
    ax.set(
        aspect='equal',
        xlabel='Real',
        ylabel='Imag',
    )
    return ax

def cursor(
    ax: plt.Axes,
    x: float = None,
    y: float = None,
    text: str = None,
    line_style: dict = None,
    text_style: dict = None,
):
    """Point out given coordinate with axhline and axvline."""
    if line_style is None: line_style = {}
    if text_style is None: text_style = {}

    xline, yline = None, None
    ls = dict(color='k', alpha=0.3, ls='--'); ls.update(line_style)
    if x is not None: xline = ax.axvline(x, **ls)
    if y is not None: yline = ax.axhline(y, **ls)

    txt = None
    if (x is not None) and (y is not None):
        if text is None: text = 'x={:.3f}, y={:.3f}'
        ts = dict(); ts.update(text_style)
        txt = ax.annotate(text.format(x, y), (x,y), **ts)
    elif x is not None:
        if text is None: text = 'x={}'
        ts = dict(rotation='vertical', va='top'); ts.update(text_style)
        if ts.get('va') == 'bottom':
            txt = ax.annotate(text.format(x), (x, ax.get_ylim()[0]), **ts)
        else:
            txt = ax.annotate(text.format(x), (x, ax.get_ylim()[1]), **ts)
    elif y is not None:
        if text is None: text = 'y={}'
        ts = dict(); ts.update(text_style)
        if ts.get('ha') == 'right':
            txt = ax.annotate(text.format(y), (ax.get_xlim()[1], y), **ts)
        else:
            txt = ax.annotate(text.format(y), (ax.get_xlim()[0], y), **ts)
    else:
        pass
    return xline, yline, txt

def get_norm(
    data: np.ndarray,
    cmin: float = None,
    cmax: float = None,
    symmetric: bool = False,
):
    """Get norm that works with cmap.
    
    Usage:
        norm(data) -> data in [0,1]

        cmap(norm_data) -> colors

        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax) -> a colorbar.

        norm.vmax, norm.vmin to get the value limits.

    Args:
        data: used to determine the min and max.
        cmin, cmax: if None, determined from data.
        symmetric: boolean, if True, cmin = -cmax.

    Returns:
        norm, extend_cbar, useful in creating colorbar.
    """
    vmin, vmax = np.min(data), np.max(data)

    if cmin is None:
        cmin = vmin
    if cmax is None:
        cmax = vmax

    if symmetric is True:
        cmax = max(abs(cmin), abs(cmax))
        cmin = -cmax

    if (vmin < cmin) and (vmax > cmax):
        extend_cbar = 'both'
    elif vmin < cmin:
        extend_cbar = 'min'
    elif vmax > cmax:
        extend_cbar = 'max'
    else:
        extend_cbar = 'neither'

    norm = mpl.colors.Normalize(cmin, cmax)
    return norm, extend_cbar


def multiple_formatter(denominator=2, number=np.pi, latex='\mathrm{\pi}'):
    """Format axis tick labels like: 1/2pi, pi, 3/2pi.

    Copied from https://stackoverflow.com/a/53586826
    
    Usage:
    ```
        ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
        ax.xaxis.set_major_formatter(multiple_formatter(2, np.pi))
    ```
    """
    def gcd(a, b):
        while b:
            a, b = b, a%b
        return a
    
    def _multiple_formatter(x, pos):
        den = denominator
        num = np.int(np.rint(den*x/number))
        com = gcd(num,den)
        (num,den) = (int(num/com),int(den/com))
        if den==1:
            if num==0:
                return '$0$'
            if num==1:
                return f'${latex}$'
            elif num==-1:
                return f'$-{latex}$'
            else:
                return f'${num}{latex}$'
        else:
            if num==1:
                return f'${latex}/{den}$'
            elif num==-1:
                return f'$-{latex}/{den}$'
            else:
                return f'${num}{latex}/{den}$'
    return plt.FuncFormatter(_multiple_formatter)
