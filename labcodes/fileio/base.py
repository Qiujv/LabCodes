import textwrap
from copy import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from attrs import define
from typing import Callable

from labcodes import plotter
from labcodes.fileio.misc import data_from_json, data_to_json

PATH_LEGAL = {
    "->": "→",
    "<-": "←",
    ":": ",",
    "|": "l",
    # '?': '？',
    "*": "·",
    "/": "",
    "\\": "",
    ">": "⟩",
    "<": "⟨",
}


@define(slots=False, repr=False)
class LogFile:
    df: pd.DataFrame
    conf: dict
    name: "LogName"
    indeps: list[str]
    deps: list[str]

    def __repr__(self):
        return f"<LogFile at {self.name}>"

    def plot(self, **kwargs):
        """Quick data plot."""
        if len(self.indeps) == 1:
            return self.plot1d(**kwargs)
        else:
            return self.plot2d(**kwargs)

    def plot1d(
        self,
        x_name: str | int = 0,
        y_name: str | int = 0,
        ax: plt.Axes = None,
        **kwargs,
    ):
        """Quick line plot."""
        set_ax_label: bool = False
        if ax is None:
            _, ax = plt.subplots()
            set_ax_label = True

        if isinstance(x_name, int):
            x_name = self.indeps[x_name]
        # convert y_name to list(str)
        if not isinstance(y_name, list):
            y_name = [y_name]
        if isinstance(y_name[0], int):
            y_name = [self.deps[i] for i in y_name]

        prefix = kwargs.pop("label", "")
        if len(y_name) == 1:
            labels = [str(prefix) + ""]
        else:
            labels = [str(prefix) + i for i in y_name]

        kw = dict(marker=".")
        kw.update(kwargs)

        for yn, lb in zip(y_name, labels):
            ax.plot(x_name, yn, data=self.df, label=lb, **kw)

        if set_ax_label:
            if np.size(y_name) > 1:
                ax.legend()
            ax.grid(True)
            ax.set_title(self.name.ptitle())
            ax.set_xlabel(x_name)
            ax.set_ylabel(y_name[0])
        return ax

    def plot2d(
        self,
        x_name: str | int = 0,
        y_name: str | int = 1,
        z_name: str | int = 0,
        ax: plt.Axes = None,
        plot_func: Callable = None,
        **kwargs,
    ):
        """Quick 2d plot with plotter.plot2d_auto."""
        if isinstance(x_name, int):
            x_name = self.indeps[x_name]
        if isinstance(y_name, int):
            y_name = self.indeps[y_name]
        if isinstance(z_name, int):
            z_name = self.deps[z_name]
        if plot_func is None:
            plot_func = plotter.plot2d_auto

        ax = plot_func(
            self.df, x_name=x_name, y_name=y_name, z_name=z_name, ax=ax, **kwargs
        )
        ax.set_title(self.name.as_plot_title())

        return ax

    @classmethod
    def load(cls, dir: Path, id: int) -> "LogFile":
        """Load a logfile from a .feather and a .json files."""
        dir = Path(dir)
        path = cls.find(dir, id, ".feather")
        df = pd.read_feather(path)
        conf = data_from_json(path.with_suffix(".json"))
        name = LogName.from_path(path)
        indeps = conf["indeps"]
        deps = conf["deps"]
        return cls(df=df, conf=conf, name=name, indeps=indeps, deps=deps)

    def save(self, dir: Path) -> Path:
        """Save a logfile into a .feather file and a .json files."""
        dir = Path(dir).resolve()
        p = dir / (self.name.fname() + ".no_suffix")
        self.df.to_feather(p.with_suffix(".feather"))

        conf = self.conf.copy()
        if "deps" not in conf:
            conf["deps"] = self.deps
        if "indeps" not in conf:
            conf["indeps"] = self.indeps
        data_to_json(conf, p.with_suffix(".json"))
        return p

    @staticmethod
    def find(dir, id="*", suffix=".feather", return_all=False):
        """Returns the full path of logfile by given ID."""
        dir = Path(dir)
        assert dir.exists()

        if suffix.startswith("."):
            suffix = suffix[1:]
        prn = f"#{id}, *.{suffix}"
        all_match = list(dir.glob(prn))
        if len(all_match) == 0:
            raise ValueError(f'Files like "{prn}" not found in {dir}')

        if return_all:
            return all_match
        else:
            return all_match[0]

    @classmethod
    def new(cls, dir, id=None, title=""):
        """Create an empty logfile at given dir.

        An empty .json file is created with the function call.
        """
        dir = Path(dir).resolve()

        if id is None:
            all_match = cls.find(dir, id="*", suffix="*", return_all=True)
            max_id = 0
            for p in all_match:
                id = p.stem[1:].split(", ")[0]
                if "," in id:
                    id = max([int(i) for i in id.split(",")])
                elif "-" in id:
                    id = max([int(i) for i in id.split("-")])
                else:
                    id = int(id)
                if id > max_id:
                    max_id = id

        name = LogName(dir=dir, id=max_id + 1, title=title)
        p = dir / name.as_file_name()
        p.with_suffix(".json").touch(exist_ok=False)  # Make a placeholder.
        return cls(df=None, conf=dict(), name=name, indeps=[], deps=[])


@define(slots=False)
class LogName:
    """A LogName.fname looks like: #[id], [title].suffix.

    suffix could be csv, ini, json, feather, png, jpg, svg...
    id could be '12' or '1,2,3' or '1-4'.
    """

    dir: str | Path
    id: str | int
    title: str

    def __str__(self):
        return f"#{self.id}, {self.title}"

    @property
    def folder(self) -> str:
        p = Path(self.dir)
        data_vault_path = [i for i in p.parents if not i.name.endswith('.dir')][0]
        folder = p.relative_to(data_vault_path).parts
        folder = [i.removesuffix('.dir') for i in folder]

        import socket
        folder.insert(0, socket.gethostname())
        return '/'.join(folder)

    def as_plot_title(self, width: int = 60) -> str:
        s = f"#{self.id}, {self.title}"
        s = textwrap.fill(s, width=width)

        # f = str(self.dir).replace(".dir", "")
        f = self.folder
        # f = textwrap.fill(f, width=width)

        s = f"{f}/\n{s}"
        return s

    ptitle = as_plot_title

    def as_file_name(self, suffix: str = "") -> str:
        s = f"#{self.id}, {self.title}"
        for k, v in PATH_LEGAL.items():
            s = s.replace(k, v)
        if suffix:
            s += suffix
        return s

    fname = as_file_name

    @classmethod
    def from_path(cls, p: str | Path) -> "LogName":
        p = Path(p)
        dir = p.parent
        id, title = p.stem[1:].split(", ", 1)
        return cls(dir=dir, id=id, title=title)

    def copy(self, id: str | int = None, title: str = None) -> "LogName":
        name = copy(self)
        if id is not None:
            name.id = id
        if title is not None:
            name.title = title
        return name

    def save_aside_data(self, fig: plt.Axes | plt.Figure) -> None:
        if isinstance(fig, plt.Axes):
            fig = fig.get_figure()
        fig.savefig(self.dir / (self.as_file_name() + ".png"))
