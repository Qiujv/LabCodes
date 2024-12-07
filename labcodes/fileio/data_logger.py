"""Data Logger based on pandas DataFrame and json_tricks."""

import inspect
import itertools
import warnings
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import Callable

import dpath
from labcodes.fileio.misc import data_to_json, data_from_json
import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm


def now() -> str:
    return datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S")


class DataLogger:
    def __init__(self, path: str = None):
        self.meta: dict = {}
        self._records: list[dict] = []
        self._segs: list[pd.DataFrame] = []
        self.path = path

    @cached_property
    def df(self) -> pd.DataFrame:
        self._flush_records()
        df = pd.concat(self._segs)
        if df.empty:
            warnings.warn("No data in DataLogger.")
        return df

    def _flush_records(self):
        if self._records:
            self._segs.append(pd.DataFrame.from_records(self._records))
            self._records = []

    def add_meta(self, meta: dict = None, **kwargs):
        if meta is None:
            meta = {}
        meta.update(kwargs)
        self.meta = dpath.merge(self.meta, meta)

    def add_meta_to_head(self, meta: dict = None, **kwargs):
        if meta is None:
            meta = {}
        meta.update(kwargs)
        self.meta = dpath.merge(meta, self.meta)

    def add_row(self, **kwargs):
        if all(np.isscalar(v) for v in kwargs.values()):
            self._records.append(kwargs)
        else:
            self._flush_records()
            self._segs.append(pd.DataFrame(kwargs))

        if hasattr(self, "df"):
            del self.df

    def capture(
        self,
        func: Callable[[float], dict[str, float | list[float]]],
        axes: list[float | list[float]] | dict[str, float | list[float]],
    ):
        if not isinstance(axes, dict):  # Assumes isinstance(axes, list)
            fsig = inspect.signature(func)
            axes = dict(zip(fsig.parameters.keys(), axes))

        run_axs: dict[str, list[float]] = {}
        const_axs: dict[str, float] = {}
        for k, v in axes.items():
            if np.iterable(v):
                run_axs[k] = v
            else:
                const_axs[k] = v
        self.add_meta_to_head(
            const=const_axs,
            dims={k: [min(a), max(a), len(a)] for k, a in run_axs.items()},
        )

        step_table = list(itertools.product(*run_axs.values()))

        with logging_redirect_tqdm():
            self.add_meta(capture_start_time=now())
            try:
                for step in tqdm(step_table, ncols=90):
                    step_kws = dict(zip(run_axs.keys(), step))
                    ret_kws = func(**step_kws, **const_axs)
                    self.add_row(**step_kws, **ret_kws)
            finally:
                self.add_meta(capture_stop_time=now())

    def plot(self):
        """Simple data plot."""
        import matplotlib.pyplot as plt
        from labcodes import plotter

        indeps = list(self.meta["dim"].keys())
        deps = [i for i in self.df.columns if i not in indeps]
        if len(indeps) == 1:
            _, ax = plt.subplots()
            ax.plot(indeps[0], deps[0], data=self.df)
            ax.set_xlabel(indeps[0])
            ax.set_ylabel(deps[0])
        else:
            ax = plotter.plot2d_auto(self.df, indeps[0], indeps[1], deps[0])
        return ax

    def save(self, path: str = None):
        if path is None:
            path = self.path
        if path is None:
            raise ValueError("No path provided.")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        if not self.df.empty:
            self.df.to_feather(p.with_suffix(".feather"))

        if self.meta:
            data_to_json(self.meta, p.with_suffix(".json"))

    @classmethod
    def load(cls, path: str) -> "DataLogger":
        p = Path(path)
        dlog = cls(path=p)

        if p.with_suffix(".json").exists():
            dlog = data_from_json(p.with_suffix(".json"))

        if p.with_suffix(".feather").exists():
            df = pd.read_feather(p.with_suffix(".feather"))
            dlog._segs = [df]

        return dlog


if __name__ == "__main__":

    def func(x, y):
        return {"z": x + y}
        # return {"z": x + y, "w": x - y * np.ones(5)}

    dlog = DataLogger()
    dlog.capture(func, [[1, 2, 3], [4, 5, 6]])
    # dlog.capture(func, {"x": [1, 2, 3], "y": [4, 5, 6]})
    print(dlog.df)
    # dlog.save("test.feather")
