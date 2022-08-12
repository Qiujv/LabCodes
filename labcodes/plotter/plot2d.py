"""Functions for general 2d plot.
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from labcodes.plotter import misc


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
        Axes with plot.
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
    norm, extend_cbar = misc.get_norm(z, cmin=cmin, cmax=cmax)
    cmap = plt.cm.get_cmap(cmap)
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

def plot2d_imshow(df, x_name, y_name, z_name, ax=None, cmin=None, cmax=None, 
                      colorbar=True, cmap='RdBu_r', norm=None, **kwargs):
    """Plot 2D scan data with imshow.

    Each data point displayed as a pixel in image. Faster and save more space than plot2d_collection.
    
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
        Axes with plot.
    """
    if ax is None:
        fig, ax = plt.subplots(tight_layout=True)
    else:
        fig = ax.get_figure()

    df = df.sort_values(by=[x_name, y_name])
    xuni = df[x_name].unique()
    yuni = df[y_name].unique()
    xsize = xuni.size
    ysize = yuni.size
    xmax, xmin = xuni.max(), xuni.min()
    ymax, ymin = yuni.max(), yuni.min()
    z = df[z_name].values.reshape(xsize, ysize).T
    norm, extend_cbar = misc.get_norm(z, cmin=cmin, cmax=cmax)
    cmap = plt.cm.get_cmap(cmap)
    colors = cmap(norm(z))
    img = ax.imshow(colors, cmap=cmap, extent=[xmin, xmax, ymin, ymax], origin='lower', aspect='auto')

    ax.set(
        xlabel=x_name, 
        ylabel=y_name, 
    )
    if colorbar is True:
        # Way to remove colorbar: ax.images[-1].colorbar.remove()
        cbar = fig.colorbar(img, ax=ax, label=z_name, extend=extend_cbar)
    return ax

    