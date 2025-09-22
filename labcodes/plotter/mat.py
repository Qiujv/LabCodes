"""Functions for plotting matrices."""

import math
from typing import TYPE_CHECKING, Callable, Literal, Sequence

import matplotlib.pyplot as plt
import numpy as np
from labcodes.plotter.misc import get_compact_font, get_norm, txt_effect
from matplotlib.collections import PatchCollection
from matplotlib.colors import (
    Colormap,
    ListedColormap,
    Normalize,
    hsv_to_rgb,
    rgb_to_hsv,
)
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator

if TYPE_CHECKING:
    from matplotlib.pyplot import ColorType
    from matplotlib.text import Text
    from mpl_toolkits.mplot3d.axes3d import Axes3D
    from numpy.typing import ArrayLike


def plot_mat(mat: np.ndarray, **kwargs):
    if np.iscomplexobj(mat):
        return plot_cmat(mat, **kwargs)
    else:
        return plot_rmat(mat, **kwargs)


def plot_mat_real_imag(
    mat: np.ndarray,
    ax_ax2: "tuple[plt.Axes, plt.Axes] | tuple[Axes3D, Axes3D] | None" = None,
    kind: Literal["matshow", "rect", "3d"] = "matshow",
    **kwargs,
):
    if ax_ax2 is None:
        if kind == "3d":
            w = max(get_fig_size_for_mat(mat))
            fig = plt.figure(layout="none", figsize=(2 * w, w))
            ax: "Axes3D" = fig.add_subplot(121, projection="3d")
            ax2: "Axes3D" = fig.add_subplot(122, projection="3d")
        else:
            w, h = get_fig_size_for_mat(mat)
            fig, (ax, ax2) = plt.subplots(figsize=(2 * w, h), ncols=2)
    else:
        ax, ax2 = ax_ax2
    kwargs["kind"] = kind
    col_real = plot_rmat(mat.real, ax=ax, **kwargs)
    col_imag = plot_rmat(mat.imag, ax=ax2, **kwargs)
    return col_real, col_imag


def plot_cmat(
    mat: np.ndarray,
    zmin: float = 0,
    zmax: float | None = None,
    ax: "plt.Axes | Axes3D | None" = None,
    kind: Literal["matshow", "rect", "3d"] = "matshow",
    text_above: float | None = 0.05,
    **kwargs,
):
    cmat, kws = complex2color(mat, zmin, zmax)
    kwargs.update(kws)
    if kind == "rect":
        ret = plot_mat_rect(mat, cmat.reshape(-1, 3), zmin, zmax, ax, **kwargs)
    elif kind == "3d":
        ret = plot_mat3d(np.abs(mat), cmat.reshape(-1, 3), ax, **kwargs)
    else:
        if ax is None:
            fig, ax = plt.subplots(figsize=get_fig_size_for_mat(mat))
        ret = ax.matshow(cmat, **kwargs)
    if text_above is not None:
        plot_mat_txt(
            np.where(np.abs(mat) < text_above, np.nan, np.abs(mat)), ax=ret.axes
        )
    return ret


def plot_rmat(
    mat: np.ndarray,
    zmin: float | None = None,
    zmax: float | None = None,
    ax: "plt.Axes | Axes3D | None" = None,
    kind: Literal["matshow", "rect", "3d"] = "matshow",
    cmap="RdBu_r",  # "bwr" is better for bar3d
    norm=None,
    text_above: float | None = 0.05,
    **kwargs,
):
    cmap = plt.get_cmap(cmap)
    if norm is None:
        norm, _ = get_norm(mat, zmin, zmax)
    kwargs["cmap"] = cmap
    kwargs["norm"] = norm
    if kind == "rect":
        cmat = cmap(norm(mat.ravel()))
        ret = plot_mat_rect(mat, cmat, zmin or 0, zmax, ax, **kwargs)
    elif kind == "3d":
        cmat = cmap(norm(mat.ravel()))
        ret = plot_mat3d(mat, cmat, ax, **kwargs)
    else:
        if ax is None:
            fig, ax = plt.subplots(figsize=get_fig_size_for_mat(mat))
        ret = ax.matshow(mat, **kwargs)
    if text_above is not None:
        plot_mat_txt(np.where(np.abs(mat) < text_above, np.nan, mat), ax=ret.axes)
    return ret


def plot_mat_rect(
    mat: np.ndarray,
    color: "ArrayLike | Sequence[ColorType] | ColorType" = "C0",
    zmin: float = 0,  # Value below will be omit.
    zmax: float = None,  # Value above in max size.
    ax: plt.Axes = None,
    frame_only: bool = False,
    **kwargs,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=get_fig_size_for_mat(mat))
    if zmax is None:
        zmax = np.nanmax(np.abs(mat))
    size = np.abs(mat)
    size = np.where(size < zmin, np.nan, size)
    size = np.where(size > zmax, zmax, size)
    size = size / zmax * 0.9 + 0.1
    xpos, ypos = np.indices(size.shape)
    squares = [
        Rectangle((x - s / 2, y - s / 2), s, s)
        for x, y, s in zip(xpos.ravel(), ypos.ravel(), size.ravel())
        if not np.isnan(s)
    ]
    if frame_only:
        col = PatchCollection(squares, edgecolor=color, facecolor="none", **kwargs)
    else:
        col = PatchCollection(squares, edgecolor="none", facecolor=color, **kwargs)
    ax.add_collection(col)
    ax.set_xlim(-0.6, mat.shape[0] - 0.4)
    ax.set_ylim(-0.6, mat.shape[1] - 0.4)
    # ax.xaxis.set_major_locator(MultipleLocator(1))
    # ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_aspect("equal")
    ax.xaxis.tick_top()
    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_inverted(True)
    return col


def plot_mat3d(
    mat: np.ndarray,
    color: "ArrayLike | Sequence[ColorType] | ColorType" = "C0",
    ax: "Axes3D" = None,
    ztick_step: float | None = 0.25,
    frame_only: bool = False,
    **kwargs,
):
    if ax is None:
        w = max(get_fig_size_for_mat(mat))
        fig = plt.figure(layout="none", figsize=(w, w))
        ax: "Axes3D" = fig.add_subplot(projection="3d")

    if frame_only:
        kws = dict(color=(0, 0, 0, 0), edgecolor=color)
        kwargs["alpha"] = None
    else:
        kws = dict(color=color, edgecolor="white")
    kws.update(kwargs)

    bar_width = 0.6
    xpos, ypos = np.indices(mat.shape)
    xpos = xpos.T.flatten() - bar_width / 2
    ypos = ypos.T.flatten() - bar_width / 2
    zpos = np.zeros(mat.size)
    dx = dy = bar_width * np.ones(mat.size)
    dz = mat.flatten()
    col = ax.bar3d(xpos, ypos, zpos, dx, dy, dz, **kws)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    if ztick_step is not None:
        ax.set_zticks(
            np.arange(
                math.floor(dz.min() / ztick_step),
                math.ceil(dz.max() / ztick_step) + 0.1,
            )
            * ztick_step
        )
    return col


def plot_mat_txt(
    mat: np.ndarray,
    ax: "plt.Axes | Axes3D" = None,
    fmt: Callable[[float], str] = "{:.2n}".format,
    skip_nan: bool = True,
    **kwargs,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=get_fig_size_for_mat(mat))

    kws = dict(
        ha="center",
        fontsize="small",
        fontstretch="condensed",
        fontfamily=get_compact_font(),
    )
    kws.update(kwargs)

    txts: "list[Text]" = []
    xpos, ypos = np.indices(mat.shape)
    for x, y, v in zip(xpos.ravel(), ypos.ravel(), mat.ravel()):
        if skip_nan:
            if np.isnan(v):
                continue
        if hasattr(ax, "text3D"):
            t = ax.text3D(
                x, y, v, fmt(v), va="bottom", path_effects=[txt_effect()], **kws
            )
        else:
            t = ax.text(x, y, fmt(v), va="center", path_effects=[txt_effect()], **kws)
        txts.append(t)
    return txts


def get_fig_size_for_mat(
    mat: np.ndarray, max_width: float = 10, max_height: float = 10
) -> tuple[float, float]:
    """Get a suitable figure size for plotting a matrix."""
    nrows, ncols = mat.shape
    width = ncols * 0.1 + 2
    height = nrows * 0.1 + 2
    return min(width, max_width), min(height, max_height)


def complex2color(
    arr: np.ndarray,
    zmin: float = 0,  # Value below in white.
    zmax: float = None,  # Value above in most saturated color.
    cmap: "Colormap" = None,
    hue_start: float = 15,
    return_cmap_norm: bool = True,
):
    mag = np.abs(arr)
    if zmax is None:
        zmax = np.nanmax(mag)
    mag = np.clip(mag, zmin, zmax)
    mag_norm = (mag - zmin) / (zmax - zmin)
    ang = np.angle(arr, deg=True) + hue_start
    ang_norm = (ang % 360) / 360
    # HSV are values in range [0,1]
    if cmap is None:
        h = ang_norm
    else:
        _rgb = cmap(ang_norm)[..., :3]
        h = rgb_to_hsv(_rgb)[..., 0]
    s = mag_norm
    v = np.ones_like(h)
    # v = 1 - 0.2 * s**2  # Adjust value to create a gradient effect
    hsv = np.stack((h, s, v), axis=-1)
    rgb = hsv_to_rgb(hsv)
    if return_cmap_norm:
        return rgb, phase_cmap(cmap, hue_start)
    else:
        return rgb


def phase_cmap(cmap: "Colormap" = None, hue_start: float = 15, npts: int = 360):
    h = np.linspace(0, 1, npts, endpoint=False)
    h = (h + hue_start / 360) % 1.0
    if cmap is not None:
        _rgb = cmap(h)[..., :3]
        h = rgb_to_hsv(_rgb)[..., 0]
    s = np.ones_like(h)
    v = np.ones_like(h)
    hsv = np.stack((h, s, v), axis=-1)
    rgb = hsv_to_rgb(hsv)
    return {"cmap": ListedColormap(rgb), "norm": Normalize(0, 360)}


def complex_colorbar(cmap: "Colormap" = None, hue_start=15, npts=360):
    mag, ang = np.meshgrid(np.linspace(0, 1, npts), np.linspace(0, 1, npts))
    h = (ang + hue_start / 360) % 1.0
    if cmap:
        _rgb = cmap(h)[..., :3]
        h = rgb_to_hsv(_rgb)[..., 0]
    s = mag
    v = np.ones_like(s)
    # v = 1 - 0.2 * s**2  # Adjust value to create a gradient effect
    hsv = np.stack((h, s, v), axis=-1)
    rgb = hsv_to_rgb(hsv)

    fig, ax = plt.subplots(subplot_kw=dict(projection="polar"), figsize=(3, 3))
    c = ax.pcolormesh(
        ang * 2 * np.pi,
        mag,
        np.ones_like(mag),
        color=rgb.reshape(-1, 3),
        shading="auto",
    )
    c.set_rasterized(True)
    ax.set_yticks([])
    ax.set_xticks(np.linspace(0, 2 * np.pi, 4, endpoint=False))
    ax.grid(False)
    return fig


if __name__ == "__main__":
    import doctest

    doctest.testmod()
    plt.show()
