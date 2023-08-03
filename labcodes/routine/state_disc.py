"""State discrimination routines for qubit states."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestCentroid


class NCenter:
    """State discriminator based on nearest centroid.

    Examples:
    >>> np.random.seed(0)
    >>> n_pts = 500
    >>> centers = [(0, 0), (1, 1), (0, 1)]
    >>> list_pts = [np.random.multivariate_normal(center, np.eye(len(centers[0])) * 0.1, n_pts) 
    ...             for center in centers]
    >>> stater, fig = NCenter.fit(list_pts, plot=True)
    >>> stater.flags(centers)
    array([0, 1, 2])
    >>> np.array([stater.probs(pts) for pts in list_pts])
    array([[0.932, 0.006, 0.062],
           [0.004, 0.948, 0.048],
           [0.042, 0.048, 0.91 ]])
    >>> stater2 = NCenter(centers)  # With ideal centers.
    >>> np.array([stater2.probs(pts) for pts in list_pts])
    array([[0.934, 0.006, 0.06 ],
           [0.004, 0.94 , 0.056],
           [0.044, 0.044, 0.912]])
    """
    def __init__(self, centers: list[np.ndarray[float]]):
        clf = NearestCentroid()
        clf.centroids_ = np.asarray(centers)
        clf.classes_ = np.arange(len(centers))
        self._clf = clf

    @property
    def centers(self):
        return self._clf.centroids_

    @classmethod
    def fit(cls, list_points: list[np.ndarray[float]], plot: bool = False):
        clf = NearestCentroid()
        clf.fit(
            np.vstack(list_points), 
            np.repeat(np.arange(len(list_points)), [len(p) for p in list_points])
        )
        
        self = cls(clf.centroids_)
        self._clf = clf

        if plot is False: return self

        stater = self
        n_clusters = len(self.centers)

        figsize = (6,3) if len(list_points) == 2 else (8,3)
        fig, axs = plt.subplots(ncols=n_clusters, figsize=figsize, sharex=True, sharey=True)
        for i, pts in enumerate(list_points):
            axs[i].scatter(pts[:,0], pts[:,1], marker=f"${i}$", color=f'C{i}')
            axs[i].set_aspect('equal')
            axs[i].set_title(f'|{i}>')
            
        for i, pts in enumerate(list_points):
            stater.plot_regions(axs[i], label=False)
            probs = stater.probs(pts)
            for j in range(n_clusters):
                center = stater.centers[j]
                axs[i].annotate(f'p{j}{i}={probs[j]:.1%}', (center[0], center[1]), ha='center')
        return stater, fig
    
    def flags(self, points: np.ndarray[float]) -> np.ndarray[int]:
        return self._clf.predict(np.vstack(points))
    
    def probs(self, points: np.ndarray[float]) -> np.ndarray[float]:
        flags = self.flags(points)
        return probs_from_flags(flags, len(self.centers), 1)
    
    def plot_regions(self, ax: plt.Axes, label=True) -> None:
        """Plot the region of each state.

        Keeps the ax lims unchanges, so plot your data points before calling this method.
        """
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

        xx, yy = np.meshgrid(np.linspace(xmin, xmax, 201), np.linspace(ymin, ymax, 201))
        ax.imshow(
            self.flags(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape),
            interpolation="nearest",
            cmap=plt.cm.Paired,
            origin="lower",
            extent=[xmin, xmax, ymin, ymax],
        )

        if label is True:
            for i, center in enumerate(self.centers):
                ax.annotate(str(i), (center[0], center[1]))


def probs_from_flags(
    flags: np.ndarray[int], 
    nlevels: int, 
    n_qbs: int, 
    return_labels: bool = False,
):
    """Calculate probabilities from flags.

    Args:
        flags: array of int, values in range(nlevel**n_qbs).
            if ndim == 2, take the first dimension as n_qbs.
        nlevels: Number of levels for qubits, same for all qubits.
        n_qbs: Number of qubits.

    Returns:
        probs (np.ndarray): Probabilities.
        labels (list[str], optional): Labels for each probability, if return_labels is True.

    Examples:
    >>> np.random.seed(0)
    >>> n_qbs, nlevels = 3, 2
    >>> flags = [np.random.randint(0, nlevels, 50) for _ in range(n_qbs)]
    >>> probs, labels = probs_from_flags(flags, nlevels, n_qbs, return_labels=True)
    >>> dict(zip(labels, probs))
    {'000': 0.1, '001': 0.12, '010': 0.08, '011': 0.1, '100': 0.12, '101': 0.14, '110': 0.18, '111': 0.16}
    """
    flags = np.asarray(flags)
    if flags.ndim == 2:
        flags = flags_mq_from_1q(flags, nlevels)

    counts = np.bincount(flags, minlength=nlevels**n_qbs)
    probs = counts / np.sum(counts)

    if return_labels is True:
        return probs, str_from_flags(np.arange(nlevels**n_qbs), n_qbs, nlevels)
    else:
        return probs
    

def flags_mq_from_1q(
    list_flags: list[np.ndarray[int]], 
    nlevels: int, 
) -> np.ndarray[int]:
    """Convert flags from 1 qubit to multi-qubit.

    Args:
        list_flags: List of flags for each qubit.
            list of array of int, values in range(nlevels).
        nlevels: Number of levels for qubits, same for all qubits.

    Returns:
        Flags for multi-qubit, array of int in range(nlevel**n_qbs) or strings.

    Examples:
    >>> list_flags = [
    ...     [0, 0, 0, 1, 1, 1, 0, 1],
    ...     [0, 0, 1, 0, 1, 0, 1, 1],
    ...     [0, 1, 0, 0, 0, 1, 1, 1]
    ... ]
    >>> flags_mq_from_1q(list_flags, 2)
    array([0, 1, 2, 4, 6, 5, 3, 7])
    """
    n_qbs = len(list_flags)
    n_pts = len(list_flags[0])

    flags_mq = np.zeros(n_pts, dtype=int)
    for i, flags_1q in enumerate(list_flags):
        flags_mq += np.asarray(flags_1q) * nlevels ** (n_qbs - i - 1)

    return flags_mq

def str_from_flags(
    flags: np.ndarray[int], 
    n_qbs: int, 
    nlevels: int = 2,
) -> np.ndarray[str]:
    """Convert flags to str.

    Args:
        flags: Flags, array of int in range(nlevels**n_qbs).
        n_qbs: Number of qubits.
        nlevels: Number of levels for qubits, same for all qubits.

    Returns:
        str_flags: Flags, array of str.

    Examples:
    >>> n_qbs, nlevels = 3, 2
    >>> str_from_flags(np.arange(nlevels**n_qbs), n_qbs, nlevels)
    array(['000', '001', '010', '011', '100', '101', '110', '111'],
          dtype='<U3')
    """
    if isinstance(flags, int): flags = [flags]
    flags = np.asarray(flags)
    str_flags = np.zeros_like(flags, dtype=f"<U{n_qbs}")
    for i in range(nlevels ** n_qbs):
        str_flags[flags == i] = np.base_repr(i, base=nlevels).zfill(n_qbs)
    return str_flags


def flags_from_str(
    str_flags: list[str],
    nlevels: int = 2,
) -> np.ndarray[int]:
    """Convert str to flags.

    Returns:
        flags: Flags, array of int in range(nlevels**n_qbs).

    Examples:
    >>> flags_from_str(['000', '001', '010', '011', '100', '101', '110', '111'], 2)
    array([0, 1, 2, 3, 4, 5, 6, 7])
    """
    if isinstance(str_flags, str): str_flags = [str_flags]
    str_flags = np.asarray(str_flags)
    n_qbs = len(str_flags[0])
    flags = np.zeros_like(str_flags, dtype=int)
    for i in range(nlevels ** n_qbs):
        flags[str_flags == np.base_repr(i, base=nlevels).zfill(n_qbs)] = i
    return flags


if __name__ == '__main__':
    import doctest
    doctest.testmod()
