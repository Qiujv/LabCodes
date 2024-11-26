import json_tricks
from typing_extensions import deprecated


@deprecated("use json_tricks instead for better readability")
def data_to_json_numpy(data: dict, fname: str) -> str:
    """Dump data dict to json file."""
    import json_numpy

    s = json_numpy.dumps(data, indent=4)
    with open(fname, "w") as f:
        f.write(s)
    return s


@deprecated("use json_tricks instead for better readability")
def data_from_json_numpy(fname: str) -> dict:
    """Load data dict from json file."""
    import json_numpy

    with open(fname, "r") as f:
        s = f.read()
    data = json_numpy.loads(s)
    return data


def data_to_json(data: dict, fname: str) -> str:
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
