"""Functions for general 2d plot."""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

from labcodes.plotter import misc


def cut(cell_centers: list[float] | np.ndarray) -> np.ndarray:
    """Returns borders of segments with center around cell_centers.

    Accepts N-element 1d array, returns (N+1)-element 1d array.

    >>> cut([0, 2, 4])
    array([-1.,  1.,  3.,  5.])
    """
    cell_centers = np.asarray(cell_centers)
    dx = np.diff(cell_centers) / 2
    if not (np.all(dx > 0) or np.all(dx < 0)):
        warnings.warn(
            "Cell_centers is not monotonically increasing or decreasing."
            "This may lead to incorrect calculated cell."
        )
    # From https://github.com/matplotlib/matplotlib/blob/9e18a343fb58a2978a8e27df03190ed21c61c343/lib/matplotlib/axes/_axes.py#L5774
    cut = np.hstack(
        [cell_centers[0] - dx[0], cell_centers[:-1] + dx, cell_centers[-1] + dx[-1]]
    )
    return cut


def plot2d_collection(
    df: pd.DataFrame,
    x_name: str | int = 0,
    y_name: str | int = 1,
    z_name: str | int = 2,
    ax: plt.Axes = None,
    cmin: float = None,
    cmax: float = None,
    colorbar: bool = True,
    cmap: str = "RdBu_r",
    norm: Normalize = None,
    transpose: bool = False,
    **kwargs,
):
    """Plot z in color versus x and y.

    Data points are plotted as rectangular scatters with proper width and height
    to fill the space. Inspired by https://stackoverflow.com/a/16240370

    Note:
        Data points are plotted column by column, try exchange x_name and y_name
        if found the plot strange.
    """
    if isinstance(x_name, int):
        x_name = df.columns[x_name]
    if isinstance(y_name, int):
        y_name = df.columns[y_name]
    if isinstance(z_name, int):
        z_name = df.columns[z_name]

    if ax is None:
        fig, ax = plt.subplots()
        if transpose:
            ax.set_xlabel(y_name)
            ax.set_ylabel(x_name)
        else:
            ax.set_xlabel(x_name)
            ax.set_ylabel(y_name)
    else:
        fig = ax.get_figure()

    # Compute widths and heights.
    df = df[[x_name, y_name, z_name]].sort_values([x_name, y_name])  # copy df.
    # Remove entry with only 1 point and hence cannot compute height.
    df = df.groupby(x_name).filter(lambda x: len(x) > 1)

    xunic = df[x_name].unique()
    xcut = cut(xunic)
    df["width"] = df[x_name].map({x: w for x, w in zip(xunic, np.diff(xcut))})
    df["xshift"] = df[x_name].map({x: s for x, s in zip(xunic, xcut[:-1])})

    df["height"] = df.groupby(x_name)[y_name].transform(lambda y: np.diff(cut(y)))
    df["yshift"] = df.groupby(x_name)[y_name].transform(lambda y: cut(y)[:-1])

    if transpose:
        xywh = df[["yshift", "xshift", "height", "width"]].itertuples(index=False)
    else:
        xywh = df[["xshift", "yshift", "width", "height"]].itertuples(index=False)
    rects = [Rectangle((x, y), w, h) for x, y, w, h in xywh]

    z = df[z_name]
    if norm is None:
        norm, extend_cbar = misc.get_norm(z, cmin=cmin, cmax=cmax)
    else:
        extend_cbar = "neither"
    cmap = plt.get_cmap(cmap)
    colors = cmap(norm(z))

    col = PatchCollection(
        rects, facecolors=colors, cmap=cmap, norm=norm, linewidth=0, **kwargs
    )

    ax.add_collection(col)
    ax.margins(0)
    ax.autoscale_view()
    if colorbar:
        # Way to remove colorbar: ax.collections[-1].colorbar.remove()
        fig.colorbar(col, ax=ax, label=z_name, extend=extend_cbar, fraction=0.03)
    return ax


def plot2d_imshow(
    df: pd.DataFrame,
    x_name: str | int = 0,
    y_name: str | int = 1,
    z_name: str | int = 2,
    ax: plt.Axes = None,
    cmin: float = None,
    cmax: float = None,
    colorbar: bool = True,
    cmap: str = "RdBu_r",
    norm: Normalize = None,
    **kwargs,
):
    """Plot z in color versus x and y with plt.imshow, with each data as a pixel in the image.

    Faster than plot2d_collection, but assumes x, y are evenly spaced and no missing data.
    """
    if isinstance(x_name, int):
        x_name = df.columns[x_name]
    if isinstance(y_name, int):
        y_name = df.columns[y_name]
    if isinstance(z_name, int):
        z_name = df.columns[z_name]

    if ax is None:
        fig, ax = plt.subplots(tight_layout=True)
        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
    else:
        fig = ax.get_figure()

    # fill missing data with nan.
    tab = df.pivot(index=y_name, columns=x_name, values=z_name)
    z = tab.values
    dx = (tab.columns[1] - tab.columns[0]) / 2
    dy = (tab.index[1] - tab.index[0]) / 2
    extent = [
        tab.columns[0] - dx,
        tab.columns[-1] + dx,
        tab.index[0] - dy,
        tab.index[-1] + dy,
    ]
    if norm is None:
        norm, extend_cbar = misc.get_norm(z, cmin=cmin, cmax=cmax)
    else:
        extend_cbar = "neither"
    cmap = plt.get_cmap(cmap)
    colors = cmap(norm(z))
    img = ax.imshow(
        colors,
        cmap=cmap,
        extent=extent,
        origin="lower",
        aspect="auto",
        interpolation="none",
        **kwargs,
    )
    img.set_norm(norm)  # For colorbar.

    if colorbar:
        # Way to remove colorbar: ax.images[-1].colorbar.remove()
        fig.colorbar(img, ax=ax, label=z_name, extend=extend_cbar, fraction=0.03)
    return ax


def plot2d_pcolor(
    df: pd.DataFrame,
    x_name: str | int = 0,
    y_name: str | int = 1,
    z_name: str | int = 2,
    ax: plt.Axes = None,
    cmin: float = None,
    cmax: float = None,
    colorbar: bool = True,
    cmap: str = "RdBu_r",
    norm: Normalize = None,
    transpose: bool = False,
    **kwargs,
):
    """Plot z in color versus x and y with plt.pcolormesh, with each data as a pixel.

    Faster than plot2d_collection, but assumes no missing data.
    """
    if isinstance(x_name, int):
        x_name = df.columns[x_name]
    if isinstance(y_name, int):
        y_name = df.columns[y_name]
    if isinstance(z_name, int):
        z_name = df.columns[z_name]

    if ax is None:
        fig, ax = plt.subplots(tight_layout=True)
        if transpose:
            ax.set_xlabel(y_name)
            ax.set_ylabel(x_name)
        else:
            ax.set_xlabel(x_name)
            ax.set_ylabel(y_name)
    else:
        fig = ax.get_figure()

    df = df[[x_name, y_name, z_name]].sort_values([x_name, y_name])  # copy df.
    xsize = df[x_name].unique().size

    if norm is None:
        norm, extend_cbar = misc.get_norm(df[z_name].values, cmin=cmin, cmax=cmax)
    else:
        extend_cbar = "neither"

    # https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.pcolormesh.html#differences-pcolor-pcolormesh
    if transpose:
        mesh = ax.pcolormesh(
            df[y_name].values.reshape(xsize, -1),
            df[x_name].values.reshape(xsize, -1),
            df[z_name].values.reshape(xsize, -1),
            norm=norm,
            cmap=cmap,
            **kwargs,
        )
    else:
        mesh = ax.pcolormesh(
            df[x_name].values.reshape(xsize, -1),
            df[y_name].values.reshape(xsize, -1),
            df[z_name].values.reshape(xsize, -1),
            norm=norm,
            cmap=cmap,
            **kwargs,
        )
    if colorbar:
        # Way to remove colorbar: ax.images[-1].colorbar.remove()
        fig.colorbar(mesh, ax=ax, label=z_name, extend=extend_cbar, fraction=0.03)
    return ax


def plot2d_auto(
    df: pd.DataFrame,
    x_name: str | int = 0,
    y_name: str | int = 1,
    z_name: str | int = 2,
    ax: plt.Axes = None,
    cmin: float = None,
    cmax: float = None,
    colorbar: bool = True,
    cmap: str = "RdBu_r",
    norm: Normalize = None,
    verbose: bool = True,
    **kwargs,
):
    """Plot z in color versus x and y.

    Using `plot2d_collection`, `plot2d_pcolor` or `plot2d_imshow` depending on the data.

    Args:
        cmin, cmax: limit of colorbar, also by `collection.set_clim()`.
        colorbar: whether to plot colorbar.
        cmap: https://matplotlib.org/stable/users/explain/colors/colormaps.html
        norm: mpl.colors.Normalize. Scale of z axis, overriding cmin, cmax.
            if None, use linear scale with limits in data.
        **kwargs: forward to plot function.
    """
    if isinstance(x_name, int):
        x_name = df.columns[x_name]
    if isinstance(y_name, int):
        y_name = df.columns[y_name]
    if isinstance(z_name, int):
        z_name = df.columns[z_name]

    xsize = df[x_name].unique().size
    ysize = df[y_name].unique().size

    kwargs.update(
        dict(
            df=df,
            x_name=x_name,
            y_name=y_name,
            z_name=z_name,
            ax=ax,
            cmin=cmin,
            cmax=cmax,
            colorbar=colorbar,
            cmap=cmap,
            norm=norm,
        )
    )

    if len(df) == xsize * ysize:
        if verbose: print('imshow')
        return plot2d_imshow(**kwargs)
    elif len(df) % xsize == 0:
        if verbose: print('pcolor')
        return plot2d_pcolor(**kwargs)
    else:
        if verbose: print('collection')
        return plot2d_collection(**kwargs)


if __name__ == "__main__":
    import pandas as pd

    x = np.linspace(0, 1, 21)
    y = np.linspace(0, 1, 21)
    y, x = np.meshgrid(y, x)
    x = x.ravel()
    y = y.ravel()

    z = np.sin(x * 2 * np.pi) + np.cos(y * 2 * np.pi)
    y2 = x**2 + y
    df = pd.DataFrame(dict(x=x, y=y, z=z, y2=y2))
    plot2d_imshow(df.iloc[:-10, :])
    plot2d_pcolor(df, "x", "y2", "z", transpose=True)
    plot2d_collection(df.iloc[:-10, :], "x", "y2", "z", transpose=True)  # Missing data.
    plt.show()
