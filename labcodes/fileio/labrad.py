# %%
import re
from configparser import ConfigParser
from pathlib import Path

import pandas as pd
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


def read_labrad(folder, id, suffix='csv'):
    path = find(folder, id)

    ini = ConfigParser()
    ini.read(path.with_suffix('.ini'))
    conf = ini_to_dict(ini)
    indeps = list(conf['independent'].keys())
    deps = list(conf['dependent'].keys())

    df = pd.read_csv(path.with_suffix(f'.{suffix}'), names=indeps + deps)
    name = logname_from_path(path)

    return LogFile(df=df, conf=conf, name=name, indeps=indeps, deps=deps)

LabradRead = read_labrad  # for back compatibility.

def find(folder, id, return_all=False):
    """Returns the full path of Labrad datafile by given data ID."""
    folder = Path(folder)
    assert folder.exists()

    prn = f'{str(id).zfill(5)} - *'
    all_match = list(folder.glob(prn))
    if len(all_match) == 0:
        raise ValueError(f'Files like "{prn}" not found in {folder}')

    if return_all is True:
        return all_match
    else:
        return all_match[0]

def ini_to_dict(ini):
    d = dict()
    d['general'] = dict(ini['General'])
    d['general']['parameter'] = d['general'].pop('parameters')
    d['comments'] = dict(ini['Comments'])

    d['parameter'] = dict()
    for i in range(int(d['general']['parameter'])):
        sect = ini[f'Parameter {i+1}']
        d['parameter'].update({sect['label']: sect['data']})

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
    folder = path.parent
    match = re.search(r'(\d+) - (.*)%c (.*)', path.stem)
    if match:
        id, qubit, title = match.group(1), match.group(2), match.group(3)  # Index starts from 1.
        qubit = ','.join([qb[2:-2] for qb in qubit.split(', ') if qb.startswith('%v')])
    else:
        id, qubit, title = path.stem[:5], '', path.stem[8:]
    id = int(id)
    title = replace(title, ESCAPE_CHARS)
    title = f'{qubit} {title}' if qubit else title
    return LogName(folder=folder, id=id, title=title)

def browse(folder, do_print=False):
    folder = Path(folder)
    ini = ConfigParser()
    read = ini.read(folder/'session.ini')
    if read:
        conf = eval(ini['Tags']['datasets'])
        conf = {k[:5]: v for k,v in conf.items()}
    else:
        conf = {}
    ret = []
    for i, p in enumerate(folder.glob('*.csv')):
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
    folder = 'C:/Users/qiujv/OneDrive/Documents/My_DataFiles/LabRAD_test/220111 dpi test'
    lf = read_labrad(folder, 68)
    lf.plot1d()
