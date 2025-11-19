from pathlib import Path

import pandas.testing as pdt

from labcodes.fileio import read_logqbit


def test_logqbit():
    path = Path(".").resolve()  # Replace this with an appropriate path if needed.
    lf = read_logqbit(path / "3")
    lf2 = read_logqbit(path, 3)
    pdt.assert_frame_equal(lf.df, lf2.df)
    assert lf.indeps == lf2.indeps
    assert lf.deps == lf2.deps
    assert lf.name.id == "3"
    assert lf.name.title == lf2.name.title
    assert lf.conf == lf2.conf
    assert not lf.df.empty
