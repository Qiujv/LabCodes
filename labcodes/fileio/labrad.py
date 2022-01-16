# %%
import re
import textwrap
from configparser import ConfigParser
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from labcodes import plotter


def replace(text, dict):
    for k, v in dict.items():
        text = text.replace(k, v)
    return text

ESCAPE_CHARS = {  # |, >, : in filename were replaced by %v, %g, %c.
    r'%v': '|',
    r'%g': '>',
    r'%c': ':',
}
PATH_LEGAL = {
    '->': '→',
    '<-': '←',
    ':': ',',
    '|': 'l',
    # '?': '？',
}
ABBREV = {  # Some abbreviations.
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
    'log mag': 'mag',
    ' ': '_',
}

class LogName(object):  # Unify self.name, new_name, _get_plot_title()
    """Handle metadatas of a LabRAD logfile."""
    def __init__(self, path, **kwargs):
        path = Path(path)
        kw = self.resolve_path(path)
        kw.update(kwargs)
        self.dir, self.id, self.qubit, self.title = kw['dir'], kw['id'], kw['qubit'], kw['title']

    @staticmethod
    def resolve_path(path):
        dir = str(path.parent).replace('.dir', '')
        match = re.search(r'(\d+) - (.*)%v(.*)%g(.*)%c (.*)', path.stem)  # |, >, : in filename were replaced by %v, %g, %c.
        if match:
            id, qubit, title = match.group(1), match.group(3), match.group(5)  # Index starts from 1.
        else:
            id, qubit, title = path.stem[:5], '', path.stem[8:]
        id = int(id)
        title = replace(title, ESCAPE_CHARS)
        return dict(dir=dir, id=id, qubit=qubit, title=title)

    def __repr__(self) -> str:
        return self.to_str()

    def to_str(self, **kwargs):
        """returns 'id qubit title'."""
        kw = self.__dict__.copy()
        kw.update(kwargs)
        return f'#{kw["id"]}: {kw["qubit"]} {kw["title"]}'

    def as_plot_title(self, width=60, **kwargs):
        filled = textwrap.fill(self.to_str(**kwargs), width=width)
        title = f'{self.dir}\\\n{filled}'
        return title

    def as_file_name(self, **kwargs):
        return replace(self.to_str(**kwargs), PATH_LEGAL)


class LabradRead(object):
    def __init__(self, dir, id, **kwargs):
        self.path = self.find(dir, id, suffix='csv')
        self.ini = self.load_ini(self.path.with_suffix('.ini'))
        self.conf = self.ini_to_dict(self.ini)
        self.name = LogName(self.path, **kwargs)
        self.indeps = list(self.conf['independent'].keys())
        self.deps = list(self.conf['dependent'].keys())
        self.df = pd.read_csv(self.path, names=self.indeps + self.deps)

    def plot(self, **kwargs):
        """Auto automatically choose plot1d or plot2d."""
        if len(self.indeps) == 1:
            return self.plot1d(**kwargs)
        else:
            return self.plot2d(**kwargs)

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
            title=self.name.as_plot_title(),
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
            title=self.name.as_plot_title(),
        )
        return ax

    @staticmethod
    def find(dir, id, suffix='csv'):
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

    @staticmethod
    def load_ini(path):
        """Load ini file as config."""
        conf = ConfigParser()
        conf.read(path)
        return conf

    @staticmethod
    def ini_to_dict(ini):
        d = dict()
        d['general'] = dict(ini['General'])
        d['general']['parameter'] = d['general'].pop('parameters')
        d['comments'] = dict(ini['Comments'])
        for k in ['independent', 'dependent', 'parameter']:
            d[k] = dict()
            do_replace = False if k == 'parameter' else True
            for i in range(int(d['general'][k])):
                sect = ini[f'{k.capitalize()} {i+1}']
                name = LabradRead._get_section_name(sect, do_replace)
                d[k].update({name: dict(sect)})
        return d

    @staticmethod
    def _get_section_name(sect, do_replace=False):
        name = '_'.join([sect[k] for k in ['category', 'label'] if sect.get(k)])
        if do_replace is True:
            name = name.lower()
            name = replace(name, ABBREV)

        if sect.get('units'):
            name += f'_{sect["units"]}'
        return name

if __name__ == '__main__':
    test_dir = 'C:/Users/qiujv/Downloads'
    logf = LabradRead(test_dir, 499)
    print(f'logflie "{logf.name}" loaded!')
    from pprint import pprint
    pprint(logf.conf)
    ax = logf.plot1d()
