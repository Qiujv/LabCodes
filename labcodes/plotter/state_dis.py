"""Functions for plotting single shot complex data by measurement."""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def plot_iq(data, ax=None, **kwargs):
    """Plot data on complex plane. 
    Scatter when points is few, otherwise hist2d."""
    if ax is None:
        fig, ax = plt.subplots(tight_layout=True)
    else:
        fig = ax.get_figure()

    data = np.ravel(data)

    n_pt_max = 3000
    if data.size <= n_pt_max:
        kw = dict(marker='.', alpha=0.3)
        kw.update(kwargs)
        ax.scatter(np.real(data), np.imag(data), **kw)
    else:
        kw = dict(bins=100, norm=mcolors.PowerNorm(0.5))
        kw.update(kwargs)
        ax.hist2d(np.real(data), np.imag(data), **kw)
    ax.set(
        aspect='equal',
        xlabel='Real',
        ylabel='Imag',
    )
    return ax

def plot_pdf_cdf(data, ax=None, ax2=None, bins=50, label=None):
    """Plot probability density and cumulative probability density.

    Args:
        data: array of real number, the measured values.
        bins: passed to ax.hist and then np.histogram_bin_edges.

    Returns:
        bin_edges, density, x, cumsum with size N+1, N, N, N, respectively.
    """
    if (ax is None) or (ax2 is None):
        fig, (ax, ax2) = plt.subplots(figsize=(8,4), tight_layout=True, ncols=2)
        
    density, bins, _ = ax.hist(data, density=True, bins=bins, alpha=0.6, label=label)
    ax.set(
        xlabel='Projection Position',
        ylabel='Probability Density',
    )

    # ax2.hist(data, bins=50, density=True, cumulative=True, histtype='step')
    x = [(a + b) /2 for a,b in zip(bins[:-1], bins[1:])]
    dx = np.diff(bins)
    cumsum = np.cumsum(density * dx)
    ax2.plot(x, cumsum, label=label)
    ax2.set(
        ylabel='Probability',
        ylim=(0,1),
    )
    return bins, density, x, cumsum

def plot_visibility(s0v, s1v, ax=None, ax2=None, bins=50):
    """Plot state visibility, two levels only.
    
    Args:
        s0v, s1v: array of real number, projected signals of |0> and |1> state.
        ax, ax2: matplotlib.axes, axes to plot PDF and CDF.
        bins: passed to `np.histogram_bin_edges`.

    Returns:
        ax, ax2 with PDF, CDF plot.
    """
    if ax is None:
        fig, ax = plt.subplots(tight_layout=True)
        ax2 = ax.twinx()

    # Plot the probability density and cumulative probability density.
    bins = np.histogram_bin_edges(np.array([s0v, s1v]).real, bins=bins)
    _, _, x0, cdf0 = plot_pdf_cdf(s0v, ax, ax2, label='|0>', bins=bins)
    _, _, x1, cdf1 = plot_pdf_cdf(s1v, ax, ax2, label='|1>', bins=bins)
    ax.legend(loc='upper left')

    # Plot visibility.
    visi = cdf1 - cdf0
    visi_argmax = np.argmax(visi)
    ax2.plot(x0, visi, label='|1>-|0>')
    ax2.annotate(f'best visibility = {np.max(visi):.3f}', (x0[visi_argmax], visi[visi_argmax]), 
        (0.35, 0.95), textcoords=ax2.transAxes, arrowprops=dict(arrowstyle="->"))
    ax2.legend(loc='center right')
    return ax, ax2


if __name__ == '__main__':
    from numpy.random import default_rng
    rng = default_rng()
    noise = 0.7 * rng.standard_normal(500)
    noise2 = 0.7 * rng.standard_normal(500)
    s0 = (1+noise) + 1j*(1+noise2)
    s1 = (-1+noise) + 1j*(-1+noise2)

    from labcodes import misc
    fig = plt.figure(figsize=(6,6), tight_layout=True)
    ax, ax2, ax3 = fig.add_subplot(221), fig.add_subplot(222), fig.add_subplot(212)
    ax4 = ax3.twinx()
    plot_iq(s0, ax=ax, label='|0>')  # The best plot maybe PDF contour plot with colored line.
    plot_iq(s1, ax=ax, label='|1>')
    s0, s1 = misc.phase_rotate([s0, s1])  # Must pass np.array.
    plot_iq(s0, ax=ax2, label='|0>')
    plot_iq(s1, ax=ax2, label='|1>')
    ax.legend()
    ax2.legend()
    plot_visibility(np.real(s0), np.real(s1), ax3, ax4)