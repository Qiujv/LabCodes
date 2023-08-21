import json_numpy  # TODO: consider replace this with https://github.com/mverleg/pyjson_tricks


def data_to_json(data: dict, fname: str):
    """Dump data dict to json file."""
    s = json_numpy.dumps(data, indent=4)
    with open(fname, "w") as f:
        f.write(s)
    return s


def data_from_json(fname: str):
    """Load data dict from json file."""
    with open(fname, "r") as f:
        s = f.read()
    data = json_numpy.loads(s)
    return data
