from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from labcodes.fileio.base import LogFile, LogName


def read_labber(path: str | Path) -> LogFile:
    with h5py.File(path, "r") as file:
        df = extract_dataframe(file)
        log_chs = [chn.decode() for chn in file["Log list"]["channel_name"]]
        step_chs = [chn for chn in df.columns if chn not in log_chs]
        meta = {k: list(v) for k, v in file["Tags"].attrs.items()}
        meta.update(file.attrs)
    name = LogName(path, "", meta["log_name"])
    return LogFile(df, meta, name, step_chs, log_chs)


def extract_dataframe(file: h5py.File) -> pd.DataFrame:
    dset = file["Data"]["Data"]
    df = {}
    for i, (name, info) in enumerate(file["Data"]["Channel names"]):
        name = name.decode()
        if info == b"":
            if name in df:
                raise ValueError(f"Duplicate channel name: {name}")
            df[name] = dset[:, i, :].ravel()
        elif info == b"Real":
            if np.any(df.get(name, 0).real != 0):
                raise ValueError(f"Duplicate channel name: {name}")
            df[name] = dset[:, i, :].ravel()
        elif info == b"Imaginary":
            if np.any(df.get(name, 0).imag != 0):
                raise ValueError(f"Duplicate channel name: {name}")
            df[name] = dset[:, i, :].ravel()
        else:
            raise ValueError(f"Unknown channel type: {info}")
    return pd.DataFrame(df)
