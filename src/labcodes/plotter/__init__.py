"""Data plotting routines."""

import matplotlib.pyplot as plt

from . import mat, matrix, misc

# from .mat import plot_cmat, plot_mat, plot_mat_real_imag, plot_mat_txt, plot_rmat
from .matrix import plot_complex_mat3d, plot_mat, plot_mat2d, plot_mat3d
from .misc import (
    cursor,
    get_compact_font,
    get_norm,
    list_all_matplotlib_fonts,
    multiple_formatter,
    plot_iq,
    txt_effect,
)
from .plot2d import plot2d_auto, plot2d_collection, plot2d_imshow, plot2d_pcolor

# TODO: make this an style.
# https://matplotlib.org/stable/tutorials/introductory/customizing.html#the-default-matplotlibrc-file
plt.rcParams["pdf.fonttype"] = 42  # Make saved pdf text editable.
plt.rcParams["svg.fonttype"] = "none"  # Make saved svg text editable.
plt.rcParams["savefig.facecolor"] = "w"  # Make white background of saved figure, otherwise transparent.
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["figure.autolayout"] = True  # Enable default tight_layout.
plt.rcParams["figure.figsize"] = (6, 4)
plt.rcParams["xtick.direction"] = "in"  # Change by ax.tick_param(direction='in')
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["axes.titlesize"] = "medium"
# plt.rcParams["axes.titlelocation"] = 'left'
# plt.rcParams['axes.formatter.limits'] = (-2,4)
