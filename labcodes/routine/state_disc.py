"""State discrimination routines for multi-qubit states. 

Contains:
    StateDis_KMeans: State Discriminator based on KMeans clustering. For 1 qubit states.
    flags_mq_from_1q: Convert flags from 1 qubit to multi-qubit.
    probs_from_flags: Calculate probabilities from flags.
"""

from typing import Union
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster


def flags_mq_from_1q(list_flags: list, nlevels: int, return_str: bool = False):
    """Convert flags from 1 qubit to multi-qubit.

    Args:
        list_flags: List of flags for each qubit.
            list of array of int, values in range(nlevels).
        nlevels: Number of levels for qubits, same for all qubits.

    Returns:
        Flags for multi-qubit, array of int in range(nlevel**n_qbs) or str like "020".

    See also:
        probs_from_flags: Calculate probabilities from flags.

    Examples:
    ```
    # Check flags as expect.
    nlevels = 3
    n_qbs = 3

    # Generate all possible flags.
    mask = np.arange(nlevels ** n_qbs)
    np.random.shuffle(mask)  # Randomly shuffles the flags order.
    list_flags = n_qbs * [np.arange(nlevels)]
    list_flags = [q.flatten()[mask] for q in np.meshgrid(*list_flags)]

    flags_mq = flags_mq_from_1q(list_flags, nlevels)
    str_flags = flags_mq_from_1q(list_flags, nlevels, return_str=True)

    df = {f'q{i}': flags for i, flags in enumerate(list_flags)}
    df['flags_mq'] = flags_mq
    df['str_flags'] = str_flags
    df = pd.DataFrame(df)
    df
    ```
    """
    n_qbs = len(list_flags)
    n_pts = len(list_flags[0])

    flags_mq = np.zeros(n_pts, dtype=int)
    for i, flags_1q in enumerate(list_flags):
        flags_mq += flags_1q * nlevels ** (n_qbs - i - 1)

    if not return_str:
        return flags_mq

    # Covert int flags into str flags.
    str_flags = np.zeros_like(flags_mq, dtype=f"<U{n_qbs}")
    for i in range(nlevels**n_qbs):
        str_flags[flags_mq == i] = np.base_repr(i, base=nlevels).zfill(n_qbs)
    return str_flags


def probs_from_flags(
    flags: np.ndarray, nlevels: int, n_qbs: int, return_labels: bool = False
):
    """Calculate probabilities from flags.

    Args:
        flags: list of int, values in range(nlevel**n_qbs).
        nlevels: Number of levels for qubits, same for all qubits.
        n_qbs: Number of qubits.

    Returns:
        probs (np.ndarray): Probabilities.
        labels (list[str], optional): Labels for each probability, if return_labels is True.

    See also:
        flags_mq_from_1q: Convert flags from 1 qubit to multi-qubit.

    Examples:
    ```
    # # Random flags.
    # flags_mq = flags_mq_from_1q([np.random.randint(0, 2, 50) for _ in range(3)], 3)

    # Manually set flags.
    flags_mq = flags_mq_from_1q(np.array([1,0,1]).reshape((3,1)), 3)

    probs, labels = probs_from_flags(flags_mq, nlevels=3, n_qbs=3, return_labels=True)
    dict(zip(labels, probs))
    ```
    """
    counts = np.bincount(flags, minlength=nlevels**n_qbs)
    probs = counts / np.sum(counts)

    if not return_labels:
        return probs

    labels = [np.base_repr(i, base=nlevels).zfill(n_qbs) for i in range(len(probs))]
    return probs, labels


class KMeans:
    """State Discriminator based on KMeans clustering.

    Fields:
        centers: list of clustering centers, in complex number.

    Methods:
        fit: class method, construct a KMeans instance.
        flags: Calculate flags for each data point.
        probs: Calculate single qubit probabilities for each possible state.
        plot_region: Plot the region of each state.

    Examples:
    ```
    # 3-level state discrimination.
    i0 = np.random.normal(loc= 1, scale=0.5, size=500)
    q0 = np.random.normal(loc= 1, scale=0.5, size=500)
    i1 = np.random.normal(loc= 1, scale=0.5, size=500)
    q1 = np.random.normal(loc=-1, scale=0.5, size=500)
    i2 = np.random.normal(loc=-1, scale=0.5, size=500)
    q2 = np.random.normal(loc= 1, scale=0.5, size=500)
    iqs = [i0 + 1j*q0, i1 + 1j*q1, i2 + 1j*q2]
    stater, ax, ax2 = KMeans.fit(iqs, plot=True)

    stater = KMeans(centers=[1+1j, 1-1j, -1+1j])  # Give ideal centers.

    # Plot data points against regions.
    fig, ax = plt.subplots()
    for i, pts in enumerate(iqs):
        ax.scatter(pts.real, pts.imag, marker=f'${i}$')
    stater.plot_regions(ax)

    stater.flags(iqs[0]), stater.probs(iqs[1])
    ```
    """

    def __init__(self, centers:list[complex]):
        """
        Args:
            centers: list of complex numbers, looks like `[cplx0, cplx1, cplx2, ...]`.
        """
        centers = np.array(centers)
        if len(centers.shape) == 2:
            centers = self._xy_to_cplx(centers)
        self.centers:np.ndarray = centers

    @classmethod
    def fit(cls, list_points: list, plot: bool = False) -> Union['KMeans', tuple['KMeans', plt.Figure]]:
        """Find centers with sklearn.cluster.KMeans.

        Args:
            list_points: list of complex IQ points for different state preparations. 
                e.g. `[arr_c0, arr_c1, arr_c2, ...]`

        Returns:
            state discriminator with found centers.
        """
        # Find centers with sklearn.cluster.KMeans.
        n_clusters = len(list_points)
        flatten_pts = np.hstack(list_points)
        flatten_pts = np.c_[flatten_pts.real, flatten_pts.imag]
        kms = sklearn.cluster.KMeans(n_clusters=n_clusters, n_init="auto").fit(flatten_pts)
        # centers = kms.cluster_centers_  # NOTE: The order is random, DO NOT use.

        # Collect cluster centers in order as given in `list_points`.
        centers = []
        added_center_index = []
        for pts in list_points:
            # pts -> idx of the clothest center.
            icenters = kms.predict(np.c_[pts.real, pts.imag])
            # Find the center with most pts of this label in neighbourhood.
            idxs, counts = np.unique(icenters, return_counts=True)
            the_idx = idxs[np.argmax(counts)]
            center = kms.cluster_centers_[the_idx]

            # Check center and clusters are mapped one-by-one.
            if the_idx in added_center_index:
                raise ValueError(
                    (
                        "Bad data quality! Cannot tell difference between "
                        f"pts[{added_center_index.index(the_idx)}] and pts[{the_idx}]"
                    )
                )

            centers.append(center[0] + 1j * center[1])
            added_center_index.append(the_idx)

        stater = cls(centers)

        if plot is True:
            figsize = (6,3) if len(list_points) == 2 else (8,3)
            fig, axs = plt.subplots(ncols=n_clusters, figsize=figsize, sharex=True, sharey=True)
            for i, pts in enumerate(list_points):
                axs[i].scatter(pts.real, pts.imag, marker=f"${i}$", color=f'C{i}')
                axs[i].set_aspect('equal')
                axs[i].set_title(f'|{i}>')
                
            for i, pts in enumerate(list_points):
                stater.plot_regions(axs[i])
                probs = stater.probs(pts)
                for j in range(n_clusters):
                    center = stater.centers[j]
                    axs[i].annotate(f'{probs[j]:.1%}\n', (center.real, center.imag), ha='center')
            return stater, fig

        return stater

    def flags(self, points: np.ndarray) -> np.ndarray:
        """Calculate state flags from complex IQ points.

        Returns:
            `[0, 2, 1, 1, 0, ...]`
        """
        n_states = len(self.centers)
        state_flags = np.argmin(
            np.abs(points - self.centers.reshape((n_states, 1))), axis=0
        )
        return state_flags

    def probs(self, points: np.ndarray) -> np.ndarray:
        """Calculate single qubit state probabilities from complex IQ points.

        Returns:
            Array with length same as centers, e.g. `[0.5, 0.02, 0.01, 0.47]`.
        """
        flags = self.flags(points)
        counts = np.bincount(flags, minlength=len(self.centers))
        return counts / len(points)

    def plot_regions(self, ax: plt.Axes):
        """Plot regions for different states on given ax.

        Examples:
        ```
        fig, ax = plt.subplots()
        ax.scatter(points.real, points.imag)
        state_dis.plot_regions(ax)
        ```
        """
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

        xx, yy = np.meshgrid(np.linspace(xmin, xmax, 201), np.linspace(ymin, ymax, 201))
        ax.imshow(
            self.flags(xx.ravel() + 1j * yy.ravel()).reshape(xx.shape),
            interpolation="nearest",
            cmap=plt.cm.Paired,
            origin="lower",
            extent=[xmin, xmax, ymin, ymax],
        )
        for i, center in enumerate(self.centers):
            ax.annotate(str(i), (center.real, center.imag))

    @staticmethod
    def _xy_to_cplx(arr):
        """[[1,2],[3,4],...] -> [1+2j, 3+4j, ...]"""
        arr = np.array(arr)
        return arr[:,0] + 1j*arr[:,1]
    
    @staticmethod
    def _cplx_to_xy(arr):
        """[1+2j, 3+4j, ...] -> [[1,2],[3,4],...]"""
        arr = np.array(arr)
        return np.c_[arr.real, arr.imag]
