"""Functions not fitting elsewhere."""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def cursor(ax, x=None, y=None, text=None, line_style={}, text_style={}):
    """Point out given coordinate with axhline and axvline."""
    xline, yline = None, None
    ls = dict(color='k', alpha=0.3, ls='--'); ls.update(line_style)
    if x is not None: xline = ax.axvline(x, **ls)
    if y is not None: yline = ax.axhline(y, **ls)

    txt = None
    if (x is not None) and (y is not None):
        if text is None: text = 'x={}, y={}'
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

def get_norm(data, cmin=None, cmax=None, symmetric=False):
    """Get norm that work with cmap.
    
    norm(data) -> data in [0,1]
    cmap(norm_data) -> colors
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax) -> a colorbar.

    Args:
        data: np.array, used to determine the min and max.
        cmin, cmax: float, if None, determined from data.
        symmetric: boolean, if True, cmin = -cmax.

    Returns:
        norm, extend_cbar, useful in creating colorbar.
    
    Notes:
        norm.vmax, norm.vmin can get the value limits.
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

# from https://stackoverflow.com/a/53586826
def multiple_formatter(denominator=2, number=np.pi, latex='\mathrm{\pi}'):
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

class Multiple:
    def __init__(self, denominator=2, number=np.pi, latex='\pi'):
        self.denominator = denominator
        self.number = number
        self.latex = latex

    def locator(self):
        return plt.MultipleLocator(self.number / self.denominator)

    def formatter(self):
        return plt.FuncFormatter(multiple_formatter(self.denominator, self.number, self.latex))