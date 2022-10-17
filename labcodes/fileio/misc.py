import json_numpy

def data_to_json(data, fname):
    """Dump data dict to json file."""
    s = json_numpy.dumps(data, indent=4)
    with open(fname, 'w') as f:
        f.write(s)
    return s

def data_from_json(fname):
    """Load data dict from json file."""
    with open(fname, 'r') as f:
        s = f.read()
    data = json_numpy.loads(s)
    return data