"""Functions for general 2d plot."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from labcodes.plotter import misc
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle


def _shift_left(arr):
    arr = np.array(arr)
    shifted = np.hstack((arr[0]*2 - arr[1], arr[:-1]))
    return (arr + shifted) / 2

def plot2d_collection(
    df: pd.DataFrame, 
    x_name: str, 
    y_name: str, 
    z_name: str, 
    ax: plt.Axes=None, 
    cmin: float=None, 
    cmax: float=None, 
    colorbar: bool=True, 
    cmap: str='RdBu_r',
    norm=None,
    **kwargs,
):
    """Plot z in color versus x and y.

    Data points are plotted as rectangular scatters with proper width and height 
    to fill the space. Inspired by https://stackoverflow.com/a/16240370

    Note: 
        Data points are plotted column by column, try exchange x_name and y_name
        if found the plot strange.
    
    Args:
        df: pandas.DataFrame.
        x_name, y_name, z_name: str, column name of data to plot.
        ax: matplotlib.axes, where to plot the figure.
        cmin, cmax: float, set the colorbar range. Same as `collection.set_clim()`.
        colorbar: bool, whether or not to plot colorbar.
        cmap: str or Colormap.
        norm: mpl.colors.Normalize or subsclasses. Scale of z axis, overriding 
            cmin, cmax.
            if None, use linear scale with limits in data.
        **kwargs: forward to mpl.collections.PathCollection.
    """
    if ax is None:
        fig, ax = plt.subplots(tight_layout=True)
    else:
        fig = ax.get_figure()

    # Compute widths and heights.
    df = df[[x_name, y_name, z_name]].sort_values([x_name, y_name])  # Returns a copy of df.
    df = df.groupby(x_name).filter(lambda x: len(x) > 1)  # Filter out entry with only 1 point and hence cannot compute height.
    
    xunic = df[x_name].unique()
    mapping = {x: w for x, w in zip(xunic, np.gradient(xunic))}
    df['width'] = df[x_name].map(mapping)

    xshift = _shift_left(xunic)
    mapping = {x: s for x, s in zip(xunic, xshift)}
    df['xshift'] = df[x_name].map(mapping)
    
    df['height'] = df.groupby(x_name)[y_name].transform(np.gradient)
    df['yshift'] = df.groupby(x_name)[y_name].transform(_shift_left)
    rects = [Rectangle((x, y), w, h)
            for x, y, w, h in df[['xshift', 'yshift', 'width', 'height']].itertuples(index=False)]

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
    if colorbar:
        # Way to remove colorbar: ax.collections[-1].colorbar.remove()
        cbar = fig.colorbar(col, ax=ax, label=z_name, extend=extend_cbar, fraction=0.05)
    return ax

def plot2d_imshow(
    df: pd.DataFrame, 
    x_name: str, 
    y_name: str, 
    z_name: str, 
    ax: plt.Axes=None, 
    cmin: float=None, 
    cmax: float=None, 
    colorbar: bool=True, 
    cmap: str='RdBu_r',
    norm=None,
):
    """Plot z in color versus x and y with plt.imshow, with each data as a pixel in the image.

    Faster than plot2d_collection, but assumes x, y are evenly spaced and no missing data.
    
    Args:
        df: pandas.DataFrame, container of data.
        x_name, y_name, z_name: str, column name of data to plot.
        ax: matplotlib.axes, where to plot the figure.
        cmin, cmax: float, set the colorbar range. Same as `collection.set_clim()`.
        colorbar: bool, whether or not to plot colorbar.
        cmap: str or Colormap.
        norm: mpl.colors.Normalize or subsclasses. Scale of z axis, overriding 
            cmin, cmax.
            if None, use linear scale with limits in data.
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
    dx2 = (xuni[1] - xuni[0]) / 2
    dy2 = (yuni[1] - yuni[0]) / 2
    xmax, xmin = xuni.max(), xuni.min()
    ymax, ymin = yuni.max(), yuni.min()
    extent = [xmin - dx2, xmax + dx2, ymin - dy2, ymax + dy2]
    z = df[z_name].values.reshape(xsize, ysize).T
    norm, extend_cbar = misc.get_norm(z, cmin=cmin, cmax=cmax)
    cmap = plt.cm.get_cmap(cmap)
    colors = cmap(norm(z))
    img = ax.imshow(colors, cmap=cmap, extent=extent, origin='lower', aspect='auto', interpolation='none')
    img.set_norm(norm)

    ax.set(
        xlabel=x_name, 
        ylabel=y_name, 
    )
    if colorbar:
        # BUG: this color bar seems have wrong clims.
        # Way to remove colorbar: ax.images[-1].colorbar.remove()
        cbar = fig.colorbar(img, ax=ax, label=z_name, extend=extend_cbar, fraction=0.03)
    return ax

    
if __name__ == '__main__':
    import pandas as pd
    x = np.linspace(0, 1, 21)
    y = np.linspace(0, 1, 21)
    x, y = np.meshgrid(x, y)
    x = x.ravel()
    y = y.ravel()

    z = np.sin(x * 2 * np.pi) + np.cos(y * 2 * np.pi)
    df = pd.DataFrame({"x": x, "y": y, "z": z})
    plot2d_imshow(df, "x", "y", "z")
    plot2d_collection(df.iloc[:-15,], "x", "y", "z")
    plt.show()
