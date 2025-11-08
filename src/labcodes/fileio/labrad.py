"""Read LabRAD logfiles and directories."""

import logging
import re
import warnings
from configparser import ConfigParser
from functools import cached_property
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from typing_extensions import deprecated

from labcodes.fileio.base import LogFile, LogName

logger = logging.getLogger(__name__)


def read_labrad(dir: Path | str, id: int = -1) -> LogFile:
    """
    Read LabRAD logfile by given data ID.
    >>> lf = read_labrad('tests/data', 3)

    or by given full path to the file.
    >>> lf = read_labrad('tests/data/00003 - power shift.csv')

    >>> type(lf.df)
    <class 'pandas.core.frame.DataFrame'>
    >>> lf.plot()  # doctest: +SKIP
    <Axes: ...
    """
    if isinstance(dir, str):
        path = Path(dir)
    elif isinstance(dir, LabradDirectory):
        path = dir.path
    else:
        path = dir

    if path.is_file():
        return read_logfile_labrad(path)
    else:
        dirc = LabradDirectory(path)
        return dirc.logfile(id)


@deprecated("`LabradRead` is deprecated, used `read_labrad` instead.")
def LabradRead(*args, **kwargs):
    return read_labrad(*args, **kwargs)


def cached_load(dir: Path | str, id: int, cache_folder: str = "./data_cache"):
    """Read LabRAD logfile by given data ID and cache it to CWD."""
    dir = Path(dir)
    dir_str = str(dir).replace(".dir", "").replace("\\", ".").replace(":", "")
    dir_str = dir_str.removeprefix("..")
    subdir = Path(cache_folder) / dir_str
    if subdir.exists():
        all_match = list(subdir.glob(f"#{id}, *.feather"))
        if all_match:
            return LogFile.load(subdir, id)
    else:
        subdir.mkdir()

    print(f"No cache found for {dir}, {id}, loading from given directory.")
    lf = read_labrad(dir, id)
    lf.save(subdir)
    return lf


class LabradDirectory:
    def __init__(self, path: Path | str):
        if isinstance(path, str):
            path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Path {path} does not exist")
        if not path.is_dir():
            raise ValueError(f"Path {path} is not a directory")

        self.path = path

    @property
    def dv_path(self) -> str:
        """Path for data vault server.

        >>> LabradDirectory('D:/Data/abc.dir/123.dir').dv_path
        '/abc/123'
        """
        return "/".join([""] + [i[:-4] for i in self.path.parts if i.endswith(".dir")])

    def find_paths(
        self,
        id: int,
        *keywords: str,
        suffix: str = ".csv",
    ) -> list[Path]:
        """Returns the full path of LabRAD datafile by given data ID.

        >>> LabradDirectory('tests/data').find_paths(3)
        [WindowsPath('tests/data/00003 - power shift.csv')]
        """
        pattern = f"{str(id).zfill(5)} - *{suffix}"

        matches = list(self.path.glob(pattern))

        if len(matches) == 0:
            logger.info(f"No matches found for {pattern}")
            return matches

        # Filter matches by keywords.
        for k in keywords:
            matches = [m for m in matches if k in m.name]

        if len(matches) == 0:
            logger.info(f"No matches found for {pattern} and keywords {keywords}")

        return matches

    @cached_property
    def ini(self) -> dict:
        config = ConfigParser()
        read = config.read(self.path / "session.ini")
        if read:
            return config
        else:
            warnings.warn(f"No config.ini found in {self.path}")
            return {}

    def browse(self, do_print=False) -> list[str]:
        tag_info: dict[str, set[Literal["star", "trash"]]] = {}
        if self.ini:
            tag_info = eval(self.ini["Tags"]["datasets"])
            # Trancate keys to id only.
            tag_info = {k[:5]: v for k, v in tag_info.items()}

        ret: list[str] = []
        for p in self.path.glob("*.csv"):
            msg = replace(p.stem, ESCAPE_CHARS)

            tags = tag_info.get(msg[:5], set())
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

    @property
    def latest_id(self):
        if self.ini:
            return int(self.ini["File System"]["counter"]) - 1
        else:
            return int(self.browse()[-1][1:6])

    def logfile(self, id: int, *keywords: str) -> LogFile:
        if id < 0:
            id = self.latest_id + id + 1
        paths = self.find_paths(id, *keywords)
        if len(paths) == 0:
            raise FileNotFoundError(
                f"No logfile with name #{id}, *{','.join(keywords)} found."
            )
        if len(paths) > 1:
            logger.warning(f"Multiple matches found for ID {id}, using the first one")

        return read_logfile_labrad(path=paths[0])


def read_logfile_labrad(path: Path | str) -> LogFile:
    """
    >>> lf = read_logfile_labrad('tests/data/00003 - power shift.csv')
    >>> type(lf.df)
    <class 'pandas.core.frame.DataFrame'>
    >>> lf.plot()  # doctest: +SKIP
    <Axes: ...
    """
    if isinstance(path, str):
        path = Path(path)

    meta = read_ini_labrad(path)
    indeps = list(meta["independent"].keys())
    deps = list(meta["dependent"].keys())
    df = pd.read_csv(path.with_suffix(".csv"), names=indeps + deps)
    name = logname_from_path(path)
    return LogFile(df=df, conf=meta, name=name, indeps=indeps, deps=deps)


def logname_from_path(path: Path) -> LogName:
    """Return LogName object from given path.

    >>> logname_from_path('tests/data/00003 - power shift.csv')
    LogName(dir=WindowsPath('tests/data'), id=3, title='power shift')
    """
    if isinstance(path, str):
        path = Path(path)

    match = re.search(r"(\d+) - (.*)%c (.*)", path.stem)
    if match:
        id, qubit, title = match.groups((1, 2, 3))  # Index starts from 1.
        qubit = ",".join([qb[2:-2] for qb in qubit.split(", ") if qb.startswith("%v")])
    else:
        id, title = path.stem.split(" - ")
        qubit = ""
    id = int(id)
    title = replace(title, ESCAPE_CHARS)
    title = f"{qubit} {title}" if qubit else title
    return LogName(dir=path.parent, id=id, title=title)


def read_ini_labrad(
    path: Path | str,
) -> dict[Literal["general", "comments", "parameter", "independent", "dependent"]]:
    if isinstance(path, str):
        path = Path(path)
    path = path.with_suffix(".ini")
    if not path.exists():
        raise FileNotFoundError(path)

    ini = ConfigParser()
    ini.read(path)

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
        sect = ini[f"Parameter {i + 1}"]
        data = sect["data"]
        # TODO: Maybe catch NameError?
        try:
            data = replace(data, _strange_numbers)
            data = eval(data, LABRAD_REG_GLOBLES)  # Parse string to proper objects.
        except:
            logging.exception(f"error parsing {sect['label']}:{sect['data']}")
        d["parameter"].update({sect["label"]: data})

    for k in ["independent", "dependent"]:
        d[k] = dict()
        for i in range(int(d["general"][k])):
            sect = ini[f"{k.capitalize()} {i + 1}"]

            name = "_".join([sect[c] for c in ["category", "label"] if sect.get(c)])
            # name = name.lower()
            name = replace(name, ABBREV)
            if sect.get("units"):
                name += "_{}".format(sect["units"])

            d[k].update({name: dict(sect)})
    return d


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


def _just_return_args(*args):
    """For LABRAD_REG_GLOBLES."""
    return args


LABRAD_REG_GLOBLES = {
    "DimensionlessArray": np.array,
    "Value": _just_return_args,
    "ValueArray": _just_return_args,
    "array": np.array,
    "uint32": int,
    "int32": int,
}
_strange_numbers = {
    "0L": "0",
    "1L": "1",
    "2L": "2",
    "3L": "3",
    "4L": "4",
    "...": '"..."',
}

if __name__ == "__main__":
    import doctest

    doctest.testmod()

    import matplotlib.pyplot as plt

    dirc = LabradDirectory("./tests/data")
    lf = dirc.logfile(3)
    lf.plot()
    plt.show()
