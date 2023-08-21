import numpy as np
import matplotlib.pyplot as plt


class MatEditor:
    """For conveniently viewing and editing matrix.

    Intended for ztalk matrix manipulation.

    Examples:
    >>> ztalk = np.array([
    ...     [1, 0.013, -0.0255, 0.0119, 0, 0, 0, 0, 0, 0.0],
    ...     [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    ...     [-0.0081, -0.016, 1, 0.0251, 0, 0, 0, 0, 0, 0],
    ...     [0, 0, 0.0569, 1, 0, 0, 0, 0, 0, 0],
    ...     [0, 0, 0, 0, 1, 0.017, -0.0169, 0.0094, 0, 0],
    ...     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    ...     [0, 0, 0, 0, -0.0058, -0.0114, 1, 0.01965, 0.014, 0.0042],
    ...     [0, 0, 0, 0, 0, 0, 0.0382, 1, 0, 0],
    ...     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    ...     [0, 0, 0, 0, 0, 0, 0.0093, 0, 0.024, 1],
    ... ])
    >>> zspace = ['Q1', 'C12', 'Q2', 'G2', 'Q4', 'C45', 'Q5', 'G5', 'C56', 'Q6']
    >>> mat = MatEditor(ztalk, zspace)
    >>> mat.show()
    <Axes: >
    >>> mat.mat[0,1]
    0.013
    >>> mat['Q1_by_C12'] = 999
    >>> mat.mat[0,1]
    999.0
    >>> mat.close()  # close the figure.
    """

    def __init__(self, mat, xlabels, ylabels=None):
        self.mat = np.array(mat)
        self.xlabels = list(xlabels)
        if ylabels is None:
            self.ylabels = list(xlabels).copy()
        else:
            self.ylabels = list(ylabels)
        self.fig = None
        self._interactive = plt.isinteractive()

    def __getitem__(self, key: str):
        xl, yl = key.split("_by_")
        xi = self.xlabels.index(xl)
        yi = self.ylabels.index(yl)
        return self.mat[xi, yi]

    def __setitem__(self, key: str, val):
        xl, yl = key.split("_by_")
        xi = self.xlabels.index(xl)
        yi = self.ylabels.index(yl)
        self.mat[xi, yi] = val

    def show(self, omit_diag: bool = True, figsize_scale: float = 0.8) -> plt.Axes:
        """Show the matrix in a figure. Diagonal terms are omitted by default."""
        vals = self.mat.copy()
        xdim, ydim = vals.shape
        xax = np.arange(xdim)
        yax = np.arange(ydim)
        xgrid, ygrid = np.meshgrid(xax, yax)
        xlabels, ylabels = self.xlabels, self.ylabels

        if omit_diag is True:
            for i in range(min(xdim, ydim)):
                vals[i, i] = 0
        mask = np.abs(vals) >= 0.2
        vmax = np.max(np.abs(vals[~mask]))

        if self.fig is None:
            fig, ax = plt.subplots(figsize=(xdim * figsize_scale, ydim * figsize_scale))
            self.fig = fig
            self._interactive = plt.isinteractive()
            plt.ion()
        else:
            fig = self.fig
            ax = fig.gca()
            ax.clear()
        ax.grid(lw=1, alpha=0.5)
        ax.set_xticks(xax)
        ax.set_yticks(yax)
        ax.set_xlim(-0.5, xdim - 0.5)
        ax.set_ylim(-0.5, ydim - 0.5)
        ax.set_xticklabels(xlabels)
        ax.set_yticklabels(ylabels)
        ax.tick_params(labelbottom=False, labeltop=True, direction="in")
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.scatter(
            xgrid[~mask],
            ygrid[~mask],
            np.abs(vals[~mask]) * 3e4,
            vals[~mask],
            cmap="bwr",
            vmin=-vmax,
            vmax=vmax,
            marker="s",
        )
        ax.scatter(xgrid[mask], ygrid[mask], 1500, "k", marker="s")
        for x, y, v in zip(xgrid.ravel(), ygrid.ravel(), vals.ravel()):
            if v == 0:
                continue
            color = "k" if abs(v) <= 0.2 else "w"
            ax.annotate(
                f"{v:.1%}\n{ylabels[y]} by {xlabels[x]}",
                (x, y),
                size="small",
                ha="center",
                va="center",
                color=color,
            )
        return ax

    def close(self):
        """Close the figure and restore the interactive state."""
        if self.fig is None:
            return
        plt.close(self.fig)
        self.fig = None
        if self._interactive is False:
            plt.ioff()
        elif self._interactive is True:
            plt.ion()


if __name__ == "__main__":
    import doctest

    doctest.testmod()
    plt.ioff()
    plt.show()
