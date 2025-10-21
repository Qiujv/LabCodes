from __future__ import annotations

from pathlib import Path

import pandas as pd
import pandas.testing as pdt
import pytest

from labcodes.fileio.base import PATH_LEGAL, LogFile, LogName


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Power_dBm": [15.0, 15.2, 15.4],
            "frequency_GHz": [4.8, 4.81, 4.82],
            "S21_log_mag_dB": [-17.4, -17.3, -17.2],
            "S21_phase_rad": [-0.74, -0.75, -0.76],
        }
    )


def test_logfile_save_and_load_roundtrip(tmp_path: Path, sample_dataframe: pd.DataFrame) -> None:
    conf = {
        "indeps": ["Power_dBm", "frequency_GHz"],
        "deps": ["S21_log_mag_dB", "S21_phase_rad"],
        "meta": "example",
    }
    name = LogName(dir=tmp_path, id=10, title="test dataset")
    log = LogFile(
        df=sample_dataframe,
        conf=conf,
        name=name,
        indeps=conf["indeps"],
        deps=conf["deps"],
    )

    placeholder = log.save(tmp_path)
    feather_path = placeholder.with_suffix(".feather")
    json_path = placeholder.with_suffix(".json")

    assert feather_path.exists()
    assert json_path.exists()

    loaded = LogFile.load(tmp_path, 10)
    pdt.assert_frame_equal(loaded.df, sample_dataframe)
    assert loaded.conf["meta"] == "example"
    assert loaded.indeps == conf["indeps"]
    assert loaded.deps == conf["deps"]
    assert loaded.name.id == "10"  # load coerces id to string via from_path
    assert loaded.name.title == "test dataset"


def test_logfile_find(tmp_path: Path) -> None:
    target = tmp_path / "#33, sample.feather"
    target.touch()

    found = LogFile.find(tmp_path, id=33)
    assert found == target

    all_matches = LogFile.find(tmp_path, id="*", suffix="feather", return_all=True)
    assert target in all_matches


def test_logfile_new_creates_placeholder(tmp_path: Path) -> None:
    existing = tmp_path / "#5, existing.csv"
    existing.touch()

    logfile = LogFile.new(tmp_path, title="fresh data")

    assert logfile.df is None
    assert logfile.indeps == []
    assert logfile.deps == []
    assert logfile.name.id == 6

    expected_json = tmp_path / (logfile.name.as_file_name() + ".json")
    assert expected_json.exists()


def test_logname_as_file_name_replaces_illegal_characters(tmp_path: Path) -> None:
    title = "rate:amplitude|scan*/value\\check"
    name = LogName(dir=tmp_path, id=1, title=title)

    sanitized = name.as_file_name()
    for illegal in PATH_LEGAL:
        assert illegal not in sanitized
    assert sanitized.startswith("#1, ")
