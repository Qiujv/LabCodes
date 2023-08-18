# LabCodes

Codes used in SUSTech superconducting quantum lab.

# Contents

`fitter`: CurveFit and BatchFit handling fitting and fit datas.

`models`: Fitting models with pre-defined initial value guess functions.

`plotter`: Data plotting routine for data generated in 2d sweep or other experiments.

`fileio`: Reading and writing data files, supports Labber, LabRAD, LTspice;

`calc`: Numerical calculator for physical models.

`tomo`: Data processing routine for quantum state tomography and process tomograpy.

`misc`: Useful functions that not fitting anywhere else.

`frog`: Codes for FROG experiments only.

# Installation

To use LabCodes,
download this repository,
go to the directory (make sure `ls` shows `setup.py`)
and run:
```powershell
pip install --editable .
```

# Documentation

All functions, classes comes with necessary documentations in their docstrings. 
Check them out with command like `help(balabala)`.
To browser the contents within, try `dir(balabala)` or `help(module_name)`.

If you are using iPython or Jupyter, try `any_object?` or `any_object??`.

It is highly recommended to read the source codes. I tried to make it easy to read.

If there is any advice or suggestions, write an issue, or pull request. 😊