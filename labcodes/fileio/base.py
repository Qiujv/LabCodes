import logging
import warnings
from pathlib import Path

import json_tricks
import matplotlib.pyplot as plt
import pandas as pd

from labcodes import plotter

logger = logging.getLogger(__name__)


class DataDirectory:
    def __init__(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(f"Path {path} does not exist")
        if not path.is_dir():
            raise ValueError(f"Path {path} is not a directory")

        self.path = path

    def find_paths(self, id: int, *keywords: str, suffix=".feather"):
        pattern = f"#{id}, *{suffix}"
        matches = list(self.path.glob(pattern))
        if len(matches) == 0:
            raise ValueError(f"No matches found for {pattern}")

        for k in keywords:
            matches = [m for m in matches if k in m.name]

        if len(matches) == 0:
            raise ValueError(f"No matches found for {pattern} and keywords {keywords}")

        return matches

    def logfile(self, id: int, *keywords: str):
        paths = self.find_paths(id, *keywords)
        if len(paths) > 1:
            logger.warning(f"Multiple matches found for ID {id}, using the first one")

        return LogFile.load(paths[0])

    @property
    def latest_id(self):
        return max(int(p.stem.split(",")[0][1:]) for p in self.find_paths("*"))

    def new(self, title: str, indeps: list[str], deps: list[str]):
        new_id = self.latest_id + 1
        save_path = self.path / FigName(new_id, title).as_file_name(".feather")
        return LogFile(
            df=pd.DataFrame(columns=indeps + deps),
            indeps=indeps,
            deps=deps,
            meta={"title": title, "indeps": indeps, "deps": deps},
            name=FigName(new_id, title),
            save_path=save_path,
        )


class LogFile:
    """A class for log files with pandas DataFrame as data and metadata.

    A log file in filesystem has following features:

    - Have a json file and a feather file with a same file name.

    - The file name is in format "#*, *"

    - There are two key 'indeps' and 'deps' in the root of json.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        indeps: list[str],
        deps: list[str],
        meta: dict = None,
        name: "FigName" = None,
        save_path: Path = None,
    ):
        if meta is None:
            meta = {}
        if name is None:
            name = FigName(0, "Untitled")
        self.df = df
        self.meta = meta
        self.name = name
        self.indeps = indeps
        self.deps = deps
        self.save_path = save_path
        self.data_to_append: list[pd.DataFrame] = []

    def __repr__(self):
        return f"<LogFile at {self.name}>"

    def plot(self, **kwargs):
        if len(self.indeps) == 1:
            return self.plot1d(**kwargs)
        else:
            return self.plot2d(**kwargs)

    def plot1d(
        self,
        x_name: str | int = 0,
        y_name: str | int | list[str | int] = 0,
        ax: plt.Axes = None,
        **kwargs,
    ):
        """Quick line plot."""
        if isinstance(x_name, int):
            x_name = self.indeps[x_name]
        if isinstance(y_name, int):
            list_y = [self.deps[y_name]]
        elif isinstance(y_name, str):
            list_y = [y_name]
        else:
            list_y = [self.deps[i] for i in y_name]

        _ax_legend = False
        if ax is None:
            _, ax = plt.subplots()
            ax.grid(True)
            ax.set_title(self.name.as_plot_title())
            ax.set_xlabel(x_name)
            if len(list_y) == 1:
                ax.set_ylabel(list_y[0])
            else:
                _ax_legend = True

        _ax_legend = kwargs.pop("legend", _ax_legend)

        prefix = kwargs.pop("label", "")
        labels = [str(prefix) + i for i in list_y]

        fit_kws = {"marker": "."}
        fit_kws.update(kwargs)

        for yn, label in zip(list_y, labels):
            ax.plot(x_name, yn, data=self.df, label=label, **fit_kws)

        if _ax_legend:
            ax.legend()
        return ax

    def plot2d(
        self,
        x_name: str | int = 0,
        y_name: str | int = 1,
        z_name: str | int = 0,
        ax: plt.Axes = None,
        **kwargs,
    ):
        """Quick 2d plot with plotter.plot2d_auto."""
        if isinstance(x_name, int):
            x_name = self.indeps[x_name]
        if isinstance(y_name, int):
            y_name = self.indeps[y_name]
        if isinstance(z_name, int):
            z_name = self.deps[z_name]

        ax = plotter.plot2d_auto(
            self.df, x_name=x_name, y_name=y_name, z_name=z_name, ax=ax, **kwargs
        )
        ax.set_title(self.name.as_plot_title())

        return ax

    def save(self, path: Path = None):
        """Save the logfile to a .feather and a .json file.

        If path.is_file(), may produce file name not in format "#*, *"
        """
        if path is None:
            path = self.save_path
        if path is None:
            warnings.warn("No save path provided, using default path", UserWarning)
            path = Path(self.name.as_file_name(".feather"))

        path = Path(path)
        if path.is_dir():  # False if not exist?
            path = path / self.name.as_file_name(".feather")

        self.df.to_feather(path.with_suffix(".feather"))

        meta = self.meta.copy()
        meta["indeps"] = self.indeps
        meta["deps"] = self.deps
        data_to_json(meta, path.with_suffix(".json"))
        return path

    @classmethod
    def load(cls, path: Path, json_path: Path = None, name: "FigName" = None):
        """Load a logfile from a .feather and a .json files."""
        path = Path(path)

        if name is None:
            name = FigName.from_str(path.stem)

        if json_path is None:
            json_path = path.with_suffix(".json")

        meta = data_from_json(json_path)

        return cls(
            df=pd.read_feather(path.with_suffix(".feather")),
            indeps=meta["indeps"],
            deps=meta["deps"],
            meta=meta,
            name=name,
            save_path=path,
        )

    @property
    def conf(self):
        warnings.warn(
            "LogFile.conf is deprecated, use LogFile.meta instead", DeprecationWarning
        )
        return self.meta

    def append(self, **rec):
        self.data_to_append.append(pd.DataFrame(rec))

    def flush(self):
        if len(self.data_to_append) > 0:
            self.df = pd.concat([self.df] + self.data_to_append, ignore_index=True)
            self.data_to_append = []

    def flush_save(self, path: Path = None):
        self.flush()
        return self.save(path)


class FigName:
    path_legal = {
        "->": "→",
        "<-": "←",
        ":": ",",
        "|": "l",
        # "?": "？",
        "*": "·",
        "/": "",
        "\\": "",
        ">": "⟩",
        "<": "⟨",
    }

    def __init__(self, id: int | str, title: str):
        self.id = id
        self.title = title

    def __str__(self):
        return f"#{self.id}, {self.title}"

    def as_plot_title(self):
        return f"#{self.id}, {self.title}"

    ptitle = as_plot_title

    def as_file_name(self, suffix=".png"):
        s = f"#{self.id}, {self.title}"

        for k, v in self.path_legal.items():
            s = s.replace(k, v)

        return s + suffix

    fname = as_file_name

    @classmethod
    def from_str(cls, s: str):
        id, title = s[1:].split(", ", 1)
        return cls(id, title)

    def copy(self, id: int | str = None, title: str = None):
        name = FigName(self.id, self.title)
        if id is not None:
            name.id = id
        if title is not None:
            name.title = title
        return name


def data_to_json(data: dict, fname: str):
    """Dump data dict to json file."""
    s = json_tricks.dumps(data, indent=4)
    with open(fname, "w") as f:
        f.write(s)
    return s


def data_from_json(fname: str) -> dict:
    """Load data dict from json file."""
    with open(fname, "r") as f:
        s = f.read()
    data = json_tricks.loads(s)
    return data


if __name__ == "__main__":
    import doctest

    doctest.testmod()
