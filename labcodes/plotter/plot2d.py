"""Functions for general 2d plot."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

from labcodes.plotter import misc


def _shift_left(arr):
    arr = np.array(arr)
    shifted = np.hstack((arr[0] * 2 - arr[1], arr[:-1]))
    return (arr + shifted) / 2


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
        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
    else:
        fig = ax.get_figure()

    # Compute widths and heights.
    df = df[[x_name, y_name, z_name]].sort_values([x_name, y_name])  # copy df.
    # Remove entry with only 1 point and hence cannot compute height.
    df = df.groupby(x_name).filter(lambda x: len(x) > 1)

    xunic = df[x_name].unique()
    mapping = {x: w for x, w in zip(xunic, np.gradient(xunic))}
    df["width"] = df[x_name].map(mapping)

    xshift = _shift_left(xunic)
    mapping = {x: s for x, s in zip(xunic, xshift)}
    df["xshift"] = df[x_name].map(mapping)

    df["height"] = df.groupby(x_name)[y_name].transform(np.gradient)
    df["yshift"] = df.groupby(x_name)[y_name].transform(_shift_left)
    rects = [
        Rectangle((x, y), w, h)
        for x, y, w, h in df[["xshift", "yshift", "width", "height"]].itertuples(
            index=False
        )
    ]

    z = df[z_name]
    if norm is None:
        norm, extend_cbar = misc.get_norm(z, cmin=cmin, cmax=cmax)
    else:
        extend_cbar = "neither"
    cmap = plt.cm.get_cmap(cmap)
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

    df = df[[x_name, y_name, z_name]].sort_values([x_name, y_name])  # copy df.
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
    if norm is None:
        norm, extend_cbar = misc.get_norm(z, cmin=cmin, cmax=cmax)
    else:
        extend_cbar = "neither"
    cmap = plt.cm.get_cmap(cmap)
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
    img.set_norm(norm)  # Messages for colorbar.

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
        return plot2d_imshow(**kwargs)
    elif len(df) % xsize == 0:
        return plot2d_pcolor(**kwargs)
    else:
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
    plot2d_imshow(df)
    plot2d_pcolor(df, "x", "y2", "z")
    plot2d_collection(df.iloc[:-10, :], "x", "y2", "z")  # Missing data.
    plt.show()
