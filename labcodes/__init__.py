import matplotlib as mpl

mpl.rcParams['pdf.fonttype'] = 42  # Make saved pdf text editable.
mpl.rcParams['savefig.facecolor'] = 'w'  # Make white background of saved figure, instead of transparent.
mpl.rcParams['savefig.dpi'] = 200
mpl.rcParams['figure.autolayout'] = True
# mpl.rcParams['axes.formatter.limits'] = (-2,4)