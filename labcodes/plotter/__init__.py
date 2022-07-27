"""Functions for quick plotting experiment datas.
"""

from labcodes.plotter.plot2d import plot2d_collection
from labcodes.plotter.mat3d import plot_mat3d, plot_complex_mat3d, plot_mat2d
from labcodes.plotter.state_dis import plot_iq, plot_visibility
from labcodes.plotter.misc import cursor


import matplotlib as mpl

mpl.rcParams['pdf.fonttype'] = 42  # Make saved pdf text editable.
mpl.rcParams['savefig.facecolor'] = 'w'  # Make white background of saved figure, instead of transparent.
mpl.rcParams['savefig.dpi'] = 200
mpl.rcParams['figure.autolayout'] = True  # Enable default tight_layout.
# mpl.rcParams['axes.titlesize'] = 'medium'
# mpl.rcParams["axes.titlelocation"] = 'left'
# mpl.rcParams['axes.formatter.limits'] = (-2,4)