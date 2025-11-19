import json
import os
import warnings
from pathlib import Path

import pandas as pd
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

from .base import LogFile, LogName

yaml = YAML()

def read_logqbit(path: Path | str, idx: int | None = None) -> LogFile:
    """
    Read LabQbit logfolder by given data ID.
    >>> lf = read_logqbit('tests/data', 3)

    or by given full path to the log folder.
    >>> lf = read_logqbit('tests/data/0')

    >>> type(lf.df)
    <class 'pandas.core.frame.DataFrame'>
    >>> lf.plot()  # doctest: +SKIP
    <Axes: ...
    """
    if idx is None:
        path = _ensure_path(path)
        idx = path.name
    elif isinstance(idx, int) and idx < 0:
        idx = latest_id(path) + 1 + idx
        path = Path(path) / str(idx)
    else:
        path = Path(path) / str(idx)

    path = _ensure_path(path)
    const_path = path / "const.yaml"
    data_path = path / "data.feather"
    meta_path = path / "metadata.json"
    if meta_path.exists():
        with open(meta_path, "r") as f:
            meta = json.load(f)
        title = meta.get("title", "")
        indeps = meta.get("plot_axes", [])
    else:
        title = ""
        indeps = []
    if data_path.exists():
        df = pd.read_feather(data_path)
        deps = [col for col in df.columns if col not in indeps]
    else:
        warnings.warn(f"{data_path=} not exist, returning empty DataFrame")
        df = pd.DataFrame()
        deps = []
    if const_path.exists():
        with open(const_path, "r") as f:
            const = yaml.load(f)
    else:
        const = CommentedMap()
    pname = LogName(dir=path.parent, id=idx, title=title)
    return LogFile(df=df, conf=const, name=pname, indeps=indeps, deps=deps)
    
        
def latest_id(parent_path: Path | str) -> int:
    parent_path = _ensure_path(parent_path)
    max_index = max(
        (
            int(entry.name)
            for entry in os.scandir(parent_path)
            if entry.is_dir() and entry.name.isdecimal()
        ),
        default=-1,
    )
    return max_index


def _ensure_path(path: Path | str, ensure_exist: bool = True) -> Path:
    if isinstance(path, str):
        path = Path(path)
    if not path.exists() and ensure_exist:
        raise FileNotFoundError(f"{path=} does not exist")
    return path
