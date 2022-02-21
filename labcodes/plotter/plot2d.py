"""Functions for general 2d plot.
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection


def plot2d_pcolormesh(df, x_name, y_name, z_name, ax=None):
    """2D pseudocolor plot with grid.
    
    Args:
        df: pandas.DataFrame, container of data.
        x_name, y_name, z_name: str, column name of data to plot.
        ax: matplotlib.axes, where to plot the figure.

    Returns:
        ax with the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(tight_layout=True)
    else:
        fig = ax.get_figure()
        
    gp = df.groupby(x_name)  # Reshape df into entries, each runs with x.
    entry_len = gp.count()[x_name].min()  # Align entry length to the shortest one. TODO: Pad all to the longest one.
    x_grid = gp[x_name].apply(lambda x: list(x[:entry_len])).values  # Array of lists.
    x_grid = np.vstack(x_grid)  # Stack to array, [[x0, x0, ..., x0], [x1, x1, ..., x1], ...]
    y_grid = gp[y_name].apply(lambda x: list(x[:entry_len])).values
    y_grid = np.vstack(y_grid)
    z_grid = gp[z_name].apply(lambda x: list(x[:entry_len])).values
    z_grid = np.vstack(z_grid)

    im = ax.pcolormesh(
        x_grid, 
        y_grid, 
        z_grid, 
        shading='nearest', 
        cmap='RdBu_r',
    )
    cbar = fig.colorbar(im, ax=ax, label=z_name)
    ax.set(xlabel=x_name, ylabel=y_name)
    return ax

def plot2d_scatter(df, x_name, y_name, z_name, ax=None, marker='s', marker_size=1):
    """2D pseudocolor plot with scatter.
    
    Args:
        df: pandas.DataFrame, container of data.
        x_name, y_name, z_name: str, column name of data to plot.
        ax: matplotlib.axes, where to plot the figure.
        marker: str, kind of scatter, 's' for square.

    Returns:
        ax with the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(tight_layout=True)
    else:
        fig = ax.get_figure()

    im = ax.scatter(
        df[x_name],
        df[y_name],
        c=df[z_name],
        s=marker_size,
        marker=marker,
        cmap='RdBu_r'
    )
    cbar = fig.colorbar(im, ax=ax, label=z_name)
    ax.margins(0)
    ax.autoscale_view()
    ax.set(
        xlabel=x_name, 
        ylabel=y_name, 
    )
    return ax

def plot2d_collection(df, x_name, y_name, z_name, ax=None, cmin=None, cmax=None, 
                      colorbar=True, cmap='RdBu_r', norm=None, **kwargs):
    """Labrad style 2D pseudocolor plot 2D scan data.

    Data points are plotted as rectangular scatters with proper width and height 
    to fill the space. Inspired by https://stackoverflow.com/a/16240370

    Note: Data points are plotted column by column, try exchange x_name and y_name
        if found the plot strange.
    
    Args:
        df: pandas.DataFrame, container of data.
        x_name, y_name, z_name: str, column name of data to plot.
        ax: matplotlib.axes, where to plot the figure.
        cmin, cmax: float, used to set colorbar range.
            Can also be achieved by `collection.set_clim()`.
        colorbar: boolean, whether or not to plot colorbar.
            True by default.
        cmap: str or Colormap.
        norm: mpl.colors.Normalize or subsclasses. Scale of z axis, overriding 
            cmin, cmax.
            if None, use linear scale with limits in data.
        **kwargs: forward to mpl.collections.PathCollection.

    Returns:
        ax with the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(tight_layout=True)
    else:
        fig = ax.get_figure()

    # Compute widths and heights.
    df = df[[x_name, y_name, z_name]].sort_values([x_name, y_name])
    indices = df[x_name].unique()
    mapping = {idx: w for idx, w in zip(indices, np.gradient(indices))}
    df['width'] = df[x_name].map(mapping)
    df = df.groupby(x_name).filter(lambda x: len(x) > 1)  # Filter out entry with only 1 point and hence cannot compute height.
    df['height'] = df.groupby(x_name)[y_name].transform(np.gradient)
    rects = [Rectangle((x - w/2, y - h/2), w, h )
            for x, y, w, h in df[[x_name, y_name, 'width', 'height']].itertuples(index=False)]

    z = df[z_name]
    if norm is None:
        norm = mpl.colors.Normalize(cmin, cmax)
    if isinstance(cmap, str):
        cmap = plt.cm.get_cmap(cmap)
    cmin, cmax = norm.vmin, norm.vmax
    zmin, zmax = z.min(), z.max()
    if cmin is None:
        cmin = zmin
    if cmax is None:
        cmax = zmax
    if (zmin < cmin) and (zmax > cmax):
        extend_cbar = 'both'
    elif zmin < cmin:
        extend_cbar = 'min'
    elif zmax > cmax:
        extend_cbar = 'max'
    else:
        extend_cbar = 'neither'
    colors = cmap(norm(z))
    col = PatchCollection(rects, facecolors=colors, cmap=cmap, norm=norm, 
                          linewidth=0, **kwargs)

    ax.add_collection(col)
    ax.margins(0)
    ax.autoscale_view()
    ax.set(
        xlabel=x_name, 
        ylabel=y_name, 
    )
    if colorbar is True:
        # Way to remove colorbar: ax.collections[-1].colorbar.remove()
        cbar = fig.colorbar(col, ax=ax, label=z_name, extend=extend_cbar)
    return ax

