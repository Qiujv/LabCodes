import logging

import json_tricks
from typing_extensions import deprecated

logger = logging.getLogger(__name__)


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
    s = json_tricks.dumps(data, indent=4, fallback_encoders=[fallback_encoder_str])
    with open(fname, "w") as f:
        f.write(s)
    return s


def data_from_json(fname: str) -> dict:
    """Load data dict from json file."""
    with open(fname, "r") as f:
        s = f.read()
    data = json_tricks.loads(s)
    return data


def fallback_encoder_str(obj, *args, is_changed=None, **kwargs):
    logger.error(f"Cannot encode {obj} to json, use str instead.")
    return str(obj)
