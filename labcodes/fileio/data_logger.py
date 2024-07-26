import inspect
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from labcodes.fileio import LogFile, LogName


# TODO: Quick temperature solution adapted from DataDirectory branch.
# TODO: maybe a joblib.parallel version.
def capture(
    func: Callable,
    axes: list[float | list[float]] | dict[str, float | list[float]],
    title: str = None,
    indeps: list[str] = None,
) -> LogFile:
    """Run a sweep on a grid over the given axes, and capture the data.

    `func` is a function that takes the axes as arguements and returns a dictionary.
    Fastest last one in the `axes`.
    if `indeps` is None, all running axes will be used as indeps.

    Axes with single element will be added to the meta data.
    """
    fsig = inspect.signature(func)
    arg_names = [i.removeprefix("_") for i in fsig.parameters.keys()]

    if isinstance(axes, dict):
        axes = [axes[k] for k in arg_names]

    # Find running axs.
    run_idx: dict[str, int] = {}
    const_axs: dict[str, float] = {}
    for i, (name, ax) in enumerate(zip(arg_names, axes)):
        if np.iterable(ax):
            run_idx[name] = i
        else:
            const_axs[name] = ax
            axes[i] = [ax]  # Make all axes iterable.
    meta = {"const": const_axs}
    meta["create_time"] = datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S")

    if title is None:
        title = func.__name__
    if indeps is None:
        indeps = list(run_idx.keys())
    else:
        indeps = [i for i in indeps if i not in const_axs]

    # Append more message to title
    dim_info = ",".join(f"{k}{len(axes[i])}" for k, i in run_idx.items())
    if dim_info:
        title = title + f" ! {dim_info}"
    fname = LogName(Path(".").resolve(), 0, title)

    step_table = list(product(*axes))
    records = []
    dfs = []
    with logging_redirect_tqdm():
        for step in tqdm(step_table, desc=title, ncols=max(90, len(title) + 50)):
            ret_kws = func(*step)
            step_vals = {k: step[i] for k, i in run_idx.items() if k in indeps}
            is_all_scalar = all(np.isscalar(i) for i in ret_kws.values())
            if is_all_scalar:
                records.append({**step_vals, **ret_kws})
            else:
                dfs.append(pd.DataFrame({**step_vals, **ret_kws}))

    if len(records) != 0 and len(dfs) != 0:
        df = pd.concat([pd.DataFrame.from_records(records)] + dfs, ignore_index=True)
    elif records:
        df = pd.DataFrame.from_records(records)
    elif dfs:
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = pd.DataFrame()

    lf = LogFile(df, meta, fname, indeps, [i for i in df.columns if i not in indeps])
    return lf


# Test functions.
if __name__ == "__main__":

    def func(a):
        return {"s21": a * np.arange(11)}

    lf = capture(
        func,
        [np.linspace(-1, 1, 21)],
        title="test",
    )
    lf.plot()
    import matplotlib.pyplot as plt

    plt.show()

if __name__ == "__main__":

    def func(a, b, c):
        return {"d": a * np.sin(b)}

    lf = capture(
        func,
        [np.linspace(-1, 1, 21), np.linspace(-np.pi, np.pi, 21), 6],
        title="orthogonal sweep",
    )
    lf.plot()
    import matplotlib.pyplot as plt

    plt.show()

if __name__ == "__main__":

    def func(zpa, df):
        f = zpa**2 + df
        p = np.cos(df)
        return {"f": f, "p": p}

    lf = capture(
        func,
        dict(zpa=np.linspace(-0.5, 1, 21), df=np.linspace(-np.pi, np.pi, 21)),
        title="non-orthogonal sweep",
        indeps=["zpa", "f"],
    )
    lf.plot()
    import matplotlib.pyplot as plt

    plt.show()
