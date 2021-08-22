"""Module containing function for quick plotting experiment datas.
"""


import numpy as np
import matplotlib.pyplot as plt
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
        cmap='coolwarm',
    )
    colorbar = fig.colorbar(im, ax=ax, label=z_name)
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
        cmap='coolwarm'
    )
    colorbar = fig.colorbar(im, ax=ax, label=z_name)
    ax.set(
        xlabel=x_name, 
        xlim=(df[x_name].min(), df[x_name].max()),
        ylabel=y_name, 
        ylim=(df[y_name].min(), df[y_name].max()),
    )
    return ax

def plot2d_scan(df, x_name, y_name, z_name, ax=None):
    """Labrad style 2D pseudocolor plot 2D scan data.

    Data points are plotted as rectangular scatters with proper width and height 
    to fill the space. Inspired by https://stackoverflow.com/a/16240370
    
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

    # Compute widths and heights.
    df = df[[x_name, y_name, z_name]].sort_values([x_name, y_name])
    indices = df[x_name].unique()
    mapping = {idx: w for idx, w in zip(indices, np.gradient(indices))}
    df['width'] = df[x_name].map(mapping)
    df['height'] = df.groupby(x_name)[y_name].transform(np.gradient)
    rects = [Rectangle((x - w/2, y - h/2), w, h )
            for x, y, w, h in df[[x_name, y_name, 'width', 'height']].itertuples(index=False)]

    z = df[z_name]
    z_norm = (z - z.min()) / (z.max() - z.min())
    cmap = plt.cm.get_cmap('coolwarm')
    col = PatchCollection(rects, facecolors=cmap(z_norm), cmap=cmap, linewidth=0)

    ax.add_collection(col)
    # ax.axis('tight')
    ax.set(
        xlabel=x_name, 
        xlim=(df[x_name].min() - df['width'].iloc[0]/2, 
              df[x_name].max() - df['width'].iloc[-1]/2),
        ylabel=y_name, 
        ylim=(df[y_name].min() - df['height'].iloc[0]/2, 
              df[y_name].max() - df['height'].iloc[-1]/2),
    )
    cbar = fig.colorbar(col, ax=ax, label=z_name)  # Also found in ax.collections[-1].colorbar
    return ax