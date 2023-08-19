"""Functions for quick plotting experiment datas.
"""

from labcodes.plotter.plot2d import plot2d_collection, plot2d_imshow
from labcodes.plotter.mat3d import plot_mat3d, plot_complex_mat3d, plot_mat2d, MatEditor, plot_mat
from labcodes.plotter.misc import plot_iq, cursor


import matplotlib as mpl
# https://matplotlib.org/stable/tutorials/introductory/customizing.html#the-default-matplotlibrc-file
mpl.rcParams['pdf.fonttype'] = 42  # Make saved pdf text editable.
mpl.rcParams['svg.fonttype'] = 'none'  # Make saved pdf text editable. Assume fonts are installed on the machine where the SVG will be viewed.
mpl.rcParams['savefig.facecolor'] = 'w'  # Make white background of saved figure, instead of transparent.
mpl.rcParams['savefig.dpi'] = 200
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['figure.autolayout'] = True  # Enable default tight_layout.
mpl.rcParams['figure.figsize'] = (6,4)
mpl.rcParams['xtick.direction'] = 'in'  # Change by ax.tick_param(direction='in')
mpl.rcParams['ytick.direction'] = 'in'
# mpl.rcParams['axes.titlesize'] = 'medium'
# mpl.rcParams["axes.titlelocation"] = 'left'
# mpl.rcParams['axes.formatter.limits'] = (-2,4)
