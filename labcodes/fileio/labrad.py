"""Module providing utilities for reading & broswer LabRAD data files."""


import logging
import re
import warnings
from configparser import ConfigParser
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from labcodes.fileio.base import LogFile, LogName

logger = logging.getLogger(__name__)

ESCAPE_CHARS = {  # |, >, : in filename were replaced by %v, %g, %c.
    r"%p": "%",
    r"%f": "/",
    r"%b": "\\",
    r"%c": ":",
    r"%a": "*",
    r"%q": "?",
    r"%r": '"',
    r"%l": "<",
    r"%g": ">",
    r"%v": "|",
}

ABBREV = {
    "pi pulse": "pi",
    "prob.": "prob",
    "|0> state": "s0",
    "|1> state": "s1",
    "|2> state": "s2",
    "|0>": "s0",
    "|1>": "s1",
    "|2>": "s2",
    "amplitude": "amp",
    "coupler bias pulse amp": "cpa",
    "coupler pulse amp": "cpa",
    "gmon pulse amp": "gpa",
    "g pulse amp": "gpa",
    "z pulse amp": "zpa",
    "readout": "ro",
    "frequency": "freq",
    "log mag": "mag",
    " ": "_",
}


def replace(text: str, dict: dict) -> str:
    """Replace substrings in text by given dict.
    
    >>> replace("a b c", {"a": "A", "b": "B"})
    'A B c'
    """
    for k, v in dict.items():
        text = text.replace(k, v)
    return text


def read_labrad(dir: Path, id: int = -1, suffix: str = None) -> LogFile:
    """Return LogFile object from Labrad datafile.

    >>> lf = read_labrad('tests/data', 3)
    >>> type(lf.df)
    <class 'pandas.core.frame.DataFrame'>
    >>> lf.plot2d()  # doctest: +ELLIPSIS
    <Axes: ...
    """
    dir = Path(dir)
    if dir.is_file():
        path = dir
    else:
        if id < 0:
            id = last_idx(dir) + id + 1
        path = find(dir, id)

    if suffix is None:
        if path.with_suffix(".csv_complete").exists():
            suffix = ".csv_complete"
        else:
            suffix = ".csv"
    if not suffix.startswith("."):
        suffix = "." + suffix

    ini = ConfigParser()
    ini.read(path.with_suffix(".ini"))
    conf = ini_to_dict(ini)
    indeps = list(conf["independent"].keys())
    deps = list(conf["dependent"].keys())

    df = pd.read_csv(path.with_suffix(suffix), names=indeps + deps)
    name = logname_from_path(path)

    return LogFile(df=df, conf=conf, name=name, indeps=indeps, deps=deps)


LabradRead = read_labrad  # for back compatibility.


def find(dir: Path, id: int, return_all: bool = False) -> Path:
    """Returns the full path of Labrad datafile by given data ID.
    
    >>> find('tests/data', 3)
    WindowsPath('tests/data/00003 - power shift.csv')
    """
    dir = Path(dir)
    if not dir.exists():
        raise FileNotFoundError(f"{dir} not exist.")

    prn = f"{str(id).zfill(5)} - *.csv"
    all_match = list(dir.glob(prn))
    if len(all_match) == 0:
        raise ValueError(f'Files like "{prn}" not found in {dir}')

    if return_all:
        return all_match
    else:
        return all_match[0]


def _just_return_args(*args):
    """For LABRAD_REG_GLOBLES."""
    return args


LABRAD_REG_GLOBLES = {
    "DimensionlessArray": np.array,
    "Value": _just_return_args,
    "ValueArray": _just_return_args,
    "array": np.array,
    "uint32": int,
}
_strange_numbers = {
    "0L": "0",
    "1L": "1",
    "2L": "2",
    "3L": "3",
    "4L": "4",
}


def ini_to_dict(ini: ConfigParser) -> dict:
    """Convert ConfigParser of a labrad config to dict.

    >>> ini = ConfigParser()
    >>> ini.read('tests/data/00003 - power shift.ini')
    ['tests/data/00003 - power shift.ini']
    >>> d = ini_to_dict(ini)
    >>> d['general']['independent']
    2
    """
    d = dict()
    d["general"] = dict(ini["General"])
    d["general"]["independent"] = int(d["general"]["independent"])
    d["general"]["dependent"] = int(d["general"]["dependent"])
    d["general"]["parameters"] = int(d["general"]["parameters"])
    d["general"]["comments"] = int(d["general"]["comments"])
    d["comments"] = dict(
        ini["Comments"]
    )  # Can be other types but I have no test example now.

    d["parameter"] = dict()
    for i in range(int(d["general"]["parameters"])):
        sect = ini[f"Parameter {i+1}"]
        data = sect["data"]
        # TODO: Maybe catch NameError?
        try:
            data = replace(data, _strange_numbers)
            data = eval(data, LABRAD_REG_GLOBLES)  # Parse string to proper objects.
        except:
            logging.exception(f'error parsing {sect["label"]}:{sect["data"]}')
        d["parameter"].update({sect["label"]: data})

    for k in ["independent", "dependent"]:
        d[k] = dict()
        for i in range(int(d["general"][k])):
            sect = ini[f"{k.capitalize()} {i+1}"]

            name = "_".join([sect[c] for c in ["category", "label"] if sect.get(c)])
            name = name.lower()
            name = replace(name, ABBREV)
            if sect.get("units"):
                name += "_{}".format(sect["units"])

            d[k].update({name: dict(sect)})
    return d


def logname_from_path(path: Path) -> LogName:
    """Return LogName object from given path.

    >>> logname_from_path(Path('tests/data/00003 - power shift.csv'))
    LogName(dir=WindowsPath('tests/data'), id=3, title='power shift')
    """
    dir = path.parent
    match = re.search(r"(\d+) - (.*)%c (.*)", path.stem)
    if match:
        id, qubit, title = (
            match.group(1),
            match.group(2),
            match.group(3),
        )  # Index starts from 1.
        qubit = ",".join([qb[2:-2] for qb in qubit.split(", ") if qb.startswith("%v")])
    else:
        id, qubit, title = path.stem[:5], "", path.stem[8:]
    id = int(id)
    title = replace(title, ESCAPE_CHARS)
    title = f"{qubit} {title}" if qubit else title
    return LogName(dir=dir, id=id, title=title)


def browse(dir: Path, do_print=False) -> List[str]:
    """Returns all datafiles in given folder. Print if do_print is True.
    
    >>> browse('tests/data')
    [' 00003 - power shift']
    """
    dir = Path(dir)
    ini = ConfigParser()
    read = ini.read(dir / "session.ini")
    if read:
        conf = eval(ini["Tags"]["datasets"])
        conf = {k[:5]: v for k, v in conf.items()}
    else:
        conf = {}
    ret = []
    for i, p in enumerate(dir.glob("*.csv")):
        msg = p.stem
        msg = replace(msg, ESCAPE_CHARS)
        tags = conf.get(msg[:5], [])
        if "trash" in tags:
            msg = "_" + msg
        elif "star" in tags:
            msg = "*" + msg
        else:
            msg = " " + msg
        if do_print:
            print(msg)
        ret.append(msg)
    return ret


def last_idx(dir: Path) -> int:
    """Returns the last index of datafile in given folder.
    
    >>> last_idx('tests/data')
    3
    """
    dir = Path(dir)
    ini = ConfigParser()
    read = ini.read(dir / "session.ini")
    if read:
        return int(ini["File System"]["counter"]) - 1
    else:
        return int(browse(dir)[-1][1:6])


def from_registry(items, **updates):
    warnings.warn("from_registry is deprecated.", DeprecationWarning)
    # if isinstance(items, dict):
    #     kws = items.copy()
    # else:
    #     kws = {k:v for k,v in items}

    kws = dict(items)
    kws.update(updates)
    return kws


def to_registry(kws, **updates):
    warnings.warn("to_registry is deprecated.", DeprecationWarning)
    kws = kws.copy()
    kws.update(updates)
    items = tuple(kws.items())
    return items


if __name__ == "__main__":
    import doctest
    import matplotlib.pyplot as plt

    doctest.testmod()
    plt.show()
