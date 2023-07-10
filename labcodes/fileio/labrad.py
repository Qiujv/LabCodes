# %%
import re
from configparser import ConfigParser
from pathlib import Path

import pandas as pd
import numpy as np
from labcodes.fileio.base import LogFile, LogName


ESCAPE_CHARS = {  # |, >, : in filename were replaced by %v, %g, %c.
    r'%v': '|',
    r'%g': '>',
    r'%c': ':',
    r'%a': '*',
    r'%f': '/',
}

ABBREV = {
    'pi pulse': 'pi',
    'prob.': 'prob',
    '|1> state': 's1',
    '|0> state': 's0',
    '|0>': 's0',
    '|1>': 's1',
    'amplitude': 'amp',
    'coupler bias pulse amp': 'cpa',
    'coupler pulse amp': 'cpa',
    'gmon pulse amp': 'gpa',
    'z pulse amp': 'zpa',
    'readout': 'ro',
    'frequency': 'freq',
    'log mag': 'mag',
    ' ': '_',
}

def replace(text, dict):
    for k, v in dict.items():
        text = text.replace(k, v)
    return text


def read_labrad(dir:Path, id:int, suffix:str=None) -> LogFile:
    path = find(dir, id)
    if suffix is None:
        if path.with_suffix('.csv_complete').exists():
            suffix = '.csv_complete'
        else:
            suffix = '.csv'
    if not suffix.startswith('.'):
        suffix = '.' + suffix

    ini = ConfigParser()
    ini.read(path.with_suffix('.ini'))
    conf = ini_to_dict(ini)
    indeps = list(conf['independent'].keys())
    deps = list(conf['dependent'].keys())

    df = pd.read_csv(path.with_suffix(suffix), names=indeps + deps)
    name = logname_from_path(path)

    return LogFile(df=df, conf=conf, name=name, indeps=indeps, deps=deps)

LabradRead = read_labrad  # for back compatibility.

def find(dir, id, return_all=False):
    """Returns the full path of Labrad datafile by given data ID."""
    dir = Path(dir)
    if not dir.exists():
        raise FileNotFoundError(f'{dir} not exist.')

    prn = f'{str(id).zfill(5)} - *'
    all_match = list(dir.glob(prn))
    if len(all_match) == 0:
        raise ValueError(f'Files like "{prn}" not found in {dir}')

    if return_all is True:
        return all_match
    else:
        return all_match[0]

def just_return_args(*args):
    return args

LABRAD_REG_GLOBLES = {
    'DimensionlessArray': np.array,
    'Value': just_return_args,
    'ValueArray': just_return_args,
    'array': np.array,
}

def ini_to_dict(ini):
    d = dict()
    d['general'] = dict(ini['General'])
    d['general']['independent'] = int(d['general']['independent'])
    d['general']['dependent'] = int(d['general']['dependent'])
    d['general']['parameters'] = int(d['general']['parameters'])
    d['general']['comments'] = int(d['general']['comments'])
    d['comments'] = dict(ini['Comments'])  # Maybe it should not be dict, but i have no test example now.

    d['parameter'] = dict()
    for i in range(int(d['general']['parameters'])):
        sect = ini[f'Parameter {i+1}']
        data = sect['data']
        # TODO: Maybe catch NameError?
        data = eval(data, LABRAD_REG_GLOBLES)  # Parse string to proper objects.
        d['parameter'].update({sect['label']: data})

    for k in ['independent', 'dependent']:
        d[k] = dict()
        for i in range(int(d['general'][k])):
            sect = ini[f'{k.capitalize()} {i+1}']

            name = '_'.join([sect[c] for c in ['category', 'label'] if sect.get(c)])
            name = name.lower()
            name = replace(name, ABBREV)
            if sect.get('units'):
                name += '_{}'.format(sect['units'])

            d[k].update({name: dict(sect)})
    return d

def logname_from_path(path):
    dir = path.parent
    match = re.search(r'(\d+) - (.*)%c (.*)', path.stem)
    if match:
        id, qubit, title = match.group(1), match.group(2), match.group(3)  # Index starts from 1.
        qubit = ','.join([qb[2:-2] for qb in qubit.split(', ') if qb.startswith('%v')])
    else:
        id, qubit, title = path.stem[:5], '', path.stem[8:]
    id = int(id)
    title = replace(title, ESCAPE_CHARS)
    title = f'{qubit} {title}' if qubit else title
    return LogName(dir=dir, id=id, title=title)

def browse(dir, do_print=False):
    dir = Path(dir)
    ini = ConfigParser()
    read = ini.read(dir/'session.ini')
    if read:
        conf = eval(ini['Tags']['datasets'])
        conf = {k[:5]: v for k,v in conf.items()}
    else:
        conf = {}
    ret = []
    for i, p in enumerate(dir.glob('*.csv')):
        msg = p.stem
        msg = replace(msg, ESCAPE_CHARS)
        tags = conf.get(msg[:5], [])
        if 'trash' in tags: msg = '_' + msg
        elif 'star' in tags: msg = '*' + msg
        else: msg = ' ' + msg
        if do_print: print(msg)
        ret.append(msg)
    return ret

def from_registry(items, **updates):
    # if isinstance(items, dict):
    #     kws = items.copy()
    # else:
    #     kws = {k:v for k,v in items}
    
    kws = dict(items)
    kws.update(updates)
    return kws

def to_registry(kws, **updates):
    kws = kws.copy()
    kws.update(updates)
    items = tuple(kws.items())
    return items

if __name__ == '__main__':
    DIR = 'C:/Users/qiujv/OneDrive/Documents/My_DataFiles/LabRAD_test/220111 dpi test'
    lf = read_labrad(DIR, 68)
    lf.plot1d()
