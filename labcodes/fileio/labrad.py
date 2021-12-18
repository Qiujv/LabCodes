# %%
import re
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path
from configparser import ConfigParser

from labcodes import plotter


def data_path(dir, id, suffix='csv'):
    """Return the full file path of Labrad datafile by given data ID.
    
    Args:
        dir: path, directory for the datafile.
        id: int, ID of the datafile.
            e.g. 12 for file named '00012 - balabala'.
        suffix: 'csv' or 'ini'.
    """
    prn = f'{str(id).zfill(5)} - *.{suffix}'
    all_match = list(Path(dir).glob(prn))
    if len(all_match) == 0:
        raise ValueError(f'Data file not found at given dir: {dir}')
    return all_match[0]

def simplify_file_name(name):
    """Convert filename into simpler one. e.g.
        '00123 - Q1, |Q2>, Q3: balabala' -> '00123 - Q2 - balabala'."""
    pattern = r'(\d+) - (.*)%v(.*)%g(.*)%c (.*)'  # |, >, : in filename were replaced by %v, %g, %c.
    match = re.search(pattern, name)
    if match is None:
        print(f'WARNING: Unknown pattern in name: {name}')
        return name
    else:
        new_name = ' - '.join([match.group(1), match.group(3), match.group(5)])  # Index starts from 1.
        new_name = new_name.replace(r'%c', ',')  # Replace ':' with ','.
        return new_name

def load_config(path):
    """Load ini file as config."""
    config = ConfigParser()
    config.read(path)
    return config

def get_param_names(config, which='Independent'):
    """Get name of independent, depedent or parameters from Labrad ini file.
    
    Args:
        config: path or configparser.ConfigParser, get by load_config(path).
        which: 'independent', 'depedent' or 'parameters'
    """
    if isinstance(config, str):
        config = load_config(config)

    num = int(config['General'][which.lower()])

    which = which.capitalize()
    if which == 'Parameters': which = which[:-1]

    names = []
    for i in range(num):
        section = config[f'{which} {i+1}']

        name = [section[k] for k in ['category', 'label'] if section.get(k)]
        if len(name) == 0:
            raise Exception(f'Cannot resolve name from section "{section.name}".')
        else:
            name = '_'.join(name)
        name = pretty_name(name)

        if section.get('units'):
            name += f'_{section["units"]}'

        names.append(name)

    return names

def pretty_name(name, abbrev=None):
    """Returns name in lowercase and words replaced according to abbrev.
    
    Args:
        name: str, to be modified.
        abbrev: dict, words to be replaced, in dict(old=new).
    """
    if abbrev is None:
        abbrev = {
            'pi pulse': 'pi',
            'prob.': 'prob',
            '|1> state': 's1',
            '|0> state': 's0',
            '|0>': 's0',
            '|1>': 's1',
            'amplitude': 'amp',
            'coupler bias pulse amp': 'cpa',
            'coupler pulse amp': 'cpa',
            'z pulse amp': 'zpa',
            'readout': 'ro',
            'frequency': 'freq',
            ' ': '_',
        }
    name = name.lower()
    for k, v in abbrev.items():
        name = name.replace(k, v)
    return name

def ini_to_dict(config):
    """Iterate over all items in ini and return them as a dict."""
    datamap = {}
    for section in config.sections():
        datamap[section] = {}
        for name, value in config.items(section):
            datamap[section].update({name:value})
    return datamap

class LabradRead(object):
    def __init__(self, dir, id):
        self.path = data_path(dir, id, suffix='csv')
        self._config = load_config(self.path.with_suffix('.ini'))
        self.name = f'{str(id).zfill(5)} - {self._config["General"]["title"]}'
        self.indeps = get_param_names(self._config, which='independent')
        self.deps = get_param_names(self._config, which='dependent')
        self.df = pd.read_csv(self.path, names=self.indeps + self.deps)

    @property
    def config(self):
        try:
            return self._config_dict
        except AttributeError:
            self._config_dict = ini_to_dict(self._config)
            return self._config_dict

    @property
    def new_name(self):
        return simplify_file_name(self.path.stem)

    def _get_plot_title(self, name=None):
        if name is None:
            name = self.name
        title = self.path.with_name(name)
        title = str(title)
        lw = 60
        title = '\n'.join([title[i:i+lw] for i in range(0, len(title), lw)])
        return title

    def plot1d(self, x_name=0, y_name=0, ax=None, **kwargs):
        """Quick line plot.
        
        Args:
            x_name, y_name: str or int, name of quantities to plot.
                if int, take self.indeps / self.deps [int].
            ax: matplotlib.axis.

        Returns:
            ax with the plot.
        """
        if ax is None:
            fig, ax = plt.subplots(tight_layout=True)
        else:
            fig = ax.get_figure()
        if isinstance(x_name, int):
            x_name = self.indeps[x_name]
        if isinstance(y_name, int):
            y_name = self.deps[y_name]
            
        kw = dict(marker='.')
        kw.update(kwargs)

        df = self.df
        ax.plot(df[x_name], df[y_name], **kw)
        ax.grid()
        ax.set(
            xlabel=x_name,
            ylabel=y_name,
            title=self._get_plot_title(),
        )
        return ax

    def plot2d(self, x_name=0, y_name=1, z_name=0, ax=None, kind='collection', **kwargs):
        """Quick 2d plot.
        
        Args:
            x_name, y_name, z_name: str or int, name of quantities to plot.
                if int, take self.indeps / self.deps [int].
            ax: matplotlib.axes.
            kind: str, 'collection', 'pcolormesh' or 'scatter'.

        Returns:
            axis with the plot.
        """
        if ax is None:
            fig, ax = plt.subplots(tight_layout=True)
        else:
            fig = ax.get_figure()
        if isinstance(x_name, int):
            x_name = self.indeps[x_name]
        if isinstance(y_name, int):
            y_name = self.indeps[y_name]
        if isinstance(z_name, int):
            z_name = self.deps[z_name]

        plot_func = getattr(plotter, f'plot2d_{kind}')
        plot_func(self.df, x_name=x_name, y_name=y_name, z_name=z_name, ax=ax, **kwargs)

        ax.set(
            xlabel=x_name,
            ylabel=y_name,
            title=self._get_plot_title(),
        )
        return ax

if __name__ == '__main__':
    logf = LabradRead('.', 13)
    print(f'logflie "{logf.name}" loaded!')