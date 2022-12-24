import json
import textwrap
from copy import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from attrs import define, field
from labcodes import plotter

PATH_LEGAL = {
    '->': '→',
    '<-': '←',
    ':': ',',
    '|': 'l',
    # '?': '？',
    '*': '·',
    '/': '',
    '\\': '',
}

@define(slots=False, repr=False)
class LogFile:
    df: field()
    conf: field()
    name: field()
    indeps: field()
    deps: field()

    def __repr__(self):
        return f'<LogFile at {self.name}>'

    def plot(self, **kwargs):
        """Quick data plot."""
        if len(self.indeps) == 1:
            return self.plot1d(**kwargs)
        else:
            return self.plot2d(**kwargs)

    def plot1d(self, x_name=0, y_name=0, ax=None, **kwargs):
        """Quick line plot."""
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        if isinstance(x_name, int):
            x_name = self.indeps[x_name]
        
        # convert y_name from int, str or list -> list(str) 
        if not isinstance(y_name, list):
            y_name = [y_name,]
        if isinstance(y_name[0], int):
            y_name = [self.deps[i] for i in y_name]

        prefix = kwargs.pop('label', '')
        if len(y_name) == 1:
            lbs = [str(prefix) + '']
        else:
            lbs = [str(prefix) + i for i in y_name]

        kw = dict(marker='.')
        kw.update(kwargs)

        df = self.df
        for yn, lb in zip(y_name, lbs):
            ax.plot(df[x_name], df[yn], label=lb, **kw)
        if np.size(y_name) > 1:
            ax.legend()

        ax.grid(True)
        ax.set(
            xlabel=x_name,
            ylabel=y_name[0],
            title=self.name.ptitle(),
        )
        return ax

    def plot2d(self, x_name=0, y_name=1, z_name=1, ax=None, kind='collection', **kwargs):
        """Quick 2d plot with plotter.plot2d_[kind]."""
        if ax is None:
            fig, ax = plt.subplots()
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

    @classmethod
    def load(cls, dir, id):
        """Load a logfile from a .feather and a .json files."""
        dir = Path(dir)
        path = cls.find(dir, id, '.feather')
        df = pd.read_feather(path)
        conf = json.load(path.with_suffix('.json'))
        name = LogName.from_path(path)
        indeps = conf['indeps']
        deps = conf['deps']
        return cls(df=df, conf=conf, name=name, indeps=indeps, deps=deps)

    def save(self, dir):
        """Save a logfile into a .feather file and a .json files."""
        dir = Path(dir).resolve()
        p = dir / self.name.fname()
        self.df.to_feather(p.with_suffix('.feather'))

        conf = self.conf.copy()
        if 'deps' not in conf:
            conf['deps'] = self.deps
        if 'indeps' not in conf:
            conf['indeps'] = self.indeps
        json.dump(conf, p.with_suffix('.json'))
        return p

    @staticmethod
    def find(dir, id='*', suffix='.feather', return_all=False):
        """Returns the full path of logfile by given ID."""
        dir = Path(dir)
        assert dir.exists()

        if suffix.startswith('.'): suffix = suffix[1:]
        prn = f'#{id}, *.{suffix}'
        all_match = list(dir.glob(prn))
        if len(all_match) == 0:
            raise ValueError(f'Files like "{prn}" not found in {dir}')

        if return_all is True:
            return all_match
        else:
            return all_match[0]
            
    @classmethod
    def new(cls, dir, id=None, title=''):
        """Create an empty logfile at given dir.
        
        An empty .json file is created with the function call.
        """
        dir = Path(dir).resolve()

        if id is None:
            all_match = cls.find(dir, id='*', suffix='*', return_all=True)
            max_id = 0
            for p in all_match:
                id = p.stem[1:].split(', ')[0]
                if ',' in id:
                    id = max([int(i) for i in id.split(',')])
                elif '-' in id:
                    id = max([int(i) for i in id.split('-')])
                else:
                    id = int(id)
                if id > max_id:
                    max_id = id

        name = LogName(dir=dir, id=max_id+1, title=title)
        p = dir / name.as_file_name()
        p.with_suffix('.json').touch(exist_ok=False)  # Make a placeholder.
        return cls(df=None, conf=dict(), name=name, indeps=[], deps=[])

    # def to_labrad(self, dir):
    #     """Save data mimic labrad. Not guaranteeing readibility by labrad.

    #     It risks damaging program files in saving files by this function instead
    #     of native API by labrad. 
    #     """
    #     pass
    
@define(slots=False, repr=False)
class LogName:
    """A LogName.fname looks like: #[id], [title].suffix.
    
    suffix could be csv, ini, json, feather, png, jpg, svg...
    id could be '12' or '1,2,3' or '1-4'.
    """
    dir: field()
    id: field()
    title: str

    def __repr__(self):
        return f'#{self.id}, {self.title}'

    def as_plot_title(self, width=60):
        s = f'#{self.id}, {self.title}'
        s = textwrap.fill(s, width=width)

        f = str(self.dir).replace('.dir', '')
        # f = textwrap.fill(f, width=width)

        s = f'{f}\\\n{s}'
        return s

    ptitle = as_plot_title

    def as_file_name(self):
        s = f'#{self.id}, {self.title}'
        for k, v in PATH_LEGAL.items():
            s = s.replace(k, v)
        return s

    fname = as_file_name

    @classmethod
    def from_path(cls, p):
        p = Path(p)
        dir = p.parent
        id, title = p.stem[1:].split(', ', 1)
        return cls(dir=dir, id=id, title=title)

    def copy(self):
        return copy(self)
