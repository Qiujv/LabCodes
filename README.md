# LabberCodes
Set of codes used in SUSTech cQED lab.

# Contents

## Analysis

Tools for data analysis, include:
  - `fileio`: Reading and writing datafiles;
  - `fitter`: Convenient fitting tools.
  - `models`: Fitting models with pre-defined initial value guess algorithm.

## Misc.

Useful function do not fit anywhere else.

`dir(labcodes.misc)` to see what it contains.

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