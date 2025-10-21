from __future__ import annotations

from pathlib import Path

import pandas as pd
import pandas.testing as pdt
import pytest

from labcodes.fileio import labrad


@pytest.fixture
def labrad_dataset(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    dataset_id = 3
    stem = f"{dataset_id:05d} - power shift"

    csv_lines = [
        "15,4.8,-17.440,-0.740",
        "16,4.82,-17.300,-0.755",
    ]
    (data_dir / f"{stem}.csv").write_text("\n".join(csv_lines))

    ini_content = """[General]
independent = 2
dependent = 2
parameters = 1
comments = 0

[Independent 1]
label = Power
units = dBm

[Independent 2]
label = frequency
units = GHz

[Dependent 1]
category = s21
label = log mag
units = dB

[Dependent 2]
category = s21
label = phase
units = rad

[Parameter 1]
label = device
data = 'example device'

[Comments]


"""
    (data_dir / f"{stem}.ini").write_text(ini_content)

    session_content = """[File System]
counter = 4

[Tags]
datasets = {'00003': {'star'}}
"""
    (data_dir / "session.ini").write_text(session_content)

    expected_columns = [
        "Power_dBm",
        "freq_GHz",
        "s21_mag_dB",
        "s21_phase_rad",
    ]
    expected_df = pd.DataFrame(
        [[15, 4.8, -17.44, -0.74], [16, 4.82, -17.3, -0.755]],
        columns=expected_columns,
    )

    return {
        "dir": data_dir,
        "dataset_id": dataset_id,
        "csv_path": data_dir / f"{stem}.csv",
        "expected_columns": expected_columns,
        "expected_df": expected_df,
    }


def test_read_logfile_labrad(labrad_dataset) -> None:
    csv_path: Path = labrad_dataset["csv_path"]
    expected_df: pd.DataFrame = labrad_dataset["expected_df"]

    logfile = labrad.read_logfile_labrad(csv_path)

    pdt.assert_frame_equal(logfile.df.reset_index(drop=True), expected_df)
    assert logfile.indeps == ["Power_dBm", "freq_GHz"]
    assert logfile.deps == ["s21_mag_dB", "s21_phase_rad"]
    assert logfile.name.id == 3
    assert logfile.name.title == "power shift"


def test_labrad_directory_logfile(labrad_dataset) -> None:
    data_dir: Path = labrad_dataset["dir"]
    dataset_id: int = labrad_dataset["dataset_id"]
    expected_df: pd.DataFrame = labrad_dataset["expected_df"]

    directory = labrad.LabradDirectory(data_dir)

    paths = directory.find_paths(dataset_id)
    assert len(paths) == 1
    assert paths[0].name.endswith(".csv")

    logfile = directory.logfile(dataset_id)
    pdt.assert_frame_equal(logfile.df.reset_index(drop=True), expected_df)


def test_read_labrad_from_directory_and_path(labrad_dataset) -> None:
    data_dir: Path = labrad_dataset["dir"]
    dataset_id: int = labrad_dataset["dataset_id"]
    expected_df: pd.DataFrame = labrad_dataset["expected_df"]

    logfile_from_dir = labrad.read_labrad(data_dir, dataset_id)
    logfile_from_path = labrad.read_labrad(labrad_dataset["csv_path"])

    pdt.assert_frame_equal(logfile_from_dir.df.reset_index(drop=True), expected_df)
    pdt.assert_frame_equal(logfile_from_path.df.reset_index(drop=True), expected_df)
