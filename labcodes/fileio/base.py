import logging
import re
import warnings
from pathlib import Path

import json_tricks
import matplotlib.pyplot as plt
import pandas as pd

from labcodes import plotter

logger = logging.getLogger(__name__)


class DataDirectory:
    def __init__(self, path: Path):
        if isinstance(path, str):
            path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Path {path} does not exist")
        if not path.is_dir():
            raise ValueError(f"Path {path} is not a directory")

        self.path = path

    def find_paths(self, id: int, *keywords: str, suffix: str = ".feather"):
        pattern = f"#{id}, *{suffix}"
        matches = list(self.path.glob(pattern))
        if len(matches) == 0:
            logger.info(f"No matches found for {pattern}")
            return matches

        for k in keywords:
            matches = [m for m in matches if k in m.name]

        if len(matches) == 0:
            logger.info(f"No matches found for {pattern} and keywords {keywords}")

        return matches

    def logfile(self, id: int, *keywords: str):
        if id < 0:
            id = self.latest_id + id + 1
        paths = self.find_paths(id, *keywords)
        if len(paths) == 0:
            raise FileNotFoundError(f"No logfile with name #{id}, *{keywords} found.")
        if len(paths) > 1:
            logger.warning(f"Multiple matches found for ID {id}, using the first one")

        return LogFile.load(paths[0])

    @property
    def latest_id(self):
        all_ids = [int(p.stem.split(",")[0][1:]) for p in self.find_paths("*")]
        if len(all_ids) == 0:
            return 0
        return max(all_ids)

    def new(self, title: str, indeps: list[str], deps: list[str]):
        new_id = self.latest_id + 1
        save_path = self.path / LogName(new_id, title).as_file_name(".feather")
        return LogFile(
            df=pd.DataFrame(columns=indeps + deps),
            indeps=indeps,
            deps=deps,
            meta={"_title": title, "_indeps": indeps, "_deps": deps},
            name=LogName(new_id, title),
            save_path=save_path,
        )


class LogFile:
    """A class for log files with pandas DataFrame as data and metadata.

    A log file in filesystem has following features:

    - Have a json file and a feather file with a same file name.

    - The file name is in format "#*, *"

    - There are two key 'indeps' and 'deps' in the root of json.

    Attributes:
        df: pandas DataFrame, for data.
        indeps: list of independent variables, for quick plotting.
        deps: list of dependent variables, for quick plotting.
        meta: metadata dict.
        name: LogName object, for name and id.
        save_path: path to the save file.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        indeps: list[str],
        deps: list[str] = None,
        meta: dict = None,
        name: "LogName" = None,
        save_path: Path = None,
    ):
        if meta is None:
            meta = {}
        if name is None:
            name = LogName(0, "Untitled")
        if isinstance(indeps, str):
            indeps = [indeps]
        if deps is None:
            deps = df.columns.drop(indeps).tolist()
        if isinstance(deps, str):
            deps = [deps]

        self.df = df
        self.meta = meta
        self.name = name
        self.indeps = indeps
        self.deps = deps
        self.save_path = save_path
        self.data_to_flush: list[pd.DataFrame] = []

    def __repr__(self):
        return f"<LogFile {self.name}>"

    def plot(self, **kwargs):
        """Quick 1d or 2d plot depending on the number of indeps."""
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
            ax.plot(self.df[x_name], self.df[yn], label=label, **fit_kws)

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
            path = Path(self.name.as_file_name(".feather"))
            logger.warning(f"No save path provided, saving to {path}")
        if isinstance(path, str):
            path = Path(path)

        if path.is_dir():  # False if not exist?
            path = path / self.name.as_file_name(".feather")

        self.df.to_feather(path.with_suffix(".feather"))

        meta = self.meta.copy()
        meta["_indeps"] = self.indeps
        meta["_deps"] = self.deps
        data_to_json(meta, path.with_suffix(".json"))
        return path

    @classmethod
    def load(cls, path: Path, json_path: Path = None):
        """Load a logfile from a .feather and a .json files."""
        if isinstance(path, str):
            path = Path(path)

        if json_path is None:
            json_path = path.with_suffix(".json")

        meta = data_from_json(json_path)

        return cls(
            df=pd.read_feather(path.with_suffix(".feather")),
            indeps=meta["_indeps"],
            deps=meta.get("_deps", None),
            meta=meta,
            name=LogName.from_str(path.stem),
            save_path=path,
        )

    @property
    def conf(self):
        warnings.warn(
            "LogFile.conf is deprecated, use LogFile.meta instead", DeprecationWarning
        )
        return self.meta

    def append(self, **rec):
        """Append a record to data.

        Buffered in the data_to_append list until flush() is called.
        """
        self.data_to_flush.append(pd.DataFrame(rec))

    def flush(self):
        """Flush the data_to_append list to the main dataframe."""
        if len(self.data_to_flush) > 0:
            self.df = pd.concat([self.df] + self.data_to_flush, ignore_index=True)
            self.data_to_flush = []


class LogName:
    """A class for id and title of log file."""

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

    def __repr__(self):
        return f'<LogName "#{self.id}, {self.title}">'

    def as_plot_title(self):
        return f"#{self.id}, {self.title}"

    ptitle = as_plot_title

    def as_file_name(self, suffix: str = ".png"):
        s = f"#{self.id}, {self.title}"

        for k, v in self.path_legal.items():
            s = s.replace(k, v)

        return s + suffix

    fname = as_file_name

    @classmethod
    def from_str(cls, s: str):
        match = re.match(r"^#(\d+), (.+)$", s)
        if match:
            id, title = match.groups()
            return cls(int(id), title)
        else:
            return cls(0, s)

    def copy(self, id: int | str = None, title: str = None):
        name = LogName(self.id, self.title)
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
