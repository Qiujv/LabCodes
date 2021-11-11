"""Functions not fitting elsewhere."""


def cursor(ax, x=None, y=None, text=None, line_style={}, text_style={}):
    """Point out given coordinate with axhline and axvline."""
    ls = dict(color='k', alpha=0.3, ls='--'); ls.update(line_style)
    if x is not None: ax.axvline(x, **ls)
    if y is not None: ax.axhline(y, **ls)

    if (x is not None) and (y is not None):
        if text is None: text = 'x={:.3e}, y={:.3e}'
        ts = dict(); ts.update(text_style)
        ax.annotate(text.format(x, y), (x,y), **ts)
    elif x is not None:
        if text is None: text = 'x={:.3e}'
        ts = dict(rotation='vertical', va='top'); ts.update(text_style)
        if ts.get('va') == 'bottom':
            ax.annotate(text.format(x), (x, ax.get_ylim()[0]), **ts)
        else:
            ax.annotate(text.format(x), (x, ax.get_ylim()[1]), **ts)
    elif y is not None:
        if text is None: text = 'y={:.3e}'
        ts = dict(); ts.update(text_style)
        if ts.get('ha') == 'right':
            ax.annotate(text.format(y), (ax.get_xlim()[1], y), **ts)
        else:
            ax.annotate(text.format(y), (ax.get_xlim()[0], y), **ts)
    else:
        pass
    return ax