# LabCodes
Codes shared in SUSTech lab.

# Contents
`fitter`: CurveFit and BatchFit handling fitting and fit datas.

`models`: Fitting models with pre-defined initial value guess algorithm.

`plotter`: Quick plotter for data generated in 2d scan, state discrimination, tomography experiments and so on.

`fileio`: Reading and writing data files, now supports Labber and LabRAD;

`misc`: Useful functions do not fit anywhere else.

`frog`: Codes for FORG experiments only.

# Installation

To use LabCodes, clone this repository, go to the directory where you clone the repository and run:
```powershell
pip install --editable .
```

### If you are using Conda

A .yml file is contained for creating an environment with all needed dependencies. First run

```powershell
conda env create --file conda_env.yml
```

and then install LabCodes with pip.