"""Module providing utilities for reading or saving data with Labber format (.hdf5).
"""

from typing import List
from pathlib import Path
from copy import deepcopy
from tqdm import tqdm
from datetime import datetime

import numpy as np
import pandas as pd

import sys
sys.path.append(r'C:\Program Files\Labber\Script')
import Labber  # pylint: disable=import-error

class Dots(pd.DataFrame):
    """pandas.DataFrame with get_dot().
    
    get_dot(**kwargs) returns dot values at specified position.
    """
    # def __init__(*args, **kwargs):
    #     super().__init__(*args, **kwargs)

    def get_dot(self, **kwargs):
        """Return dot values at specified position."""
        stepper = list(kwargs.keys())
        position = tuple(kwargs.values())
        dot = self.set_index(stepper).loc[position, :]  # TODO: deal with the 0.10000002 case.
        return dot

class MyLogfile(object):
    """Customized Labber.LogFile with more pythonic interface.

    Only rewrites attrbutes irrelavant to datas. Interacting with data should be
    implemented by subclasses.
    
    Properties:
        logfile: Labber.LogFile
        path: pathlib.Path, path of the logfile.
        step_channels: list(dict), list of step channels.
        log_channels: list(dict), list of log channels.
        ch_names: list(str), list of step channel names and log channel names.
        num_of_entries: int, number of data entries.
        user: str, the tag 'User'.
        tags: list(str), tags.
        project: str, the tag 'Project'.
        comment: str, the tag 'Comment'.
            modifying the metadatas will change the saved file dynamically.

    Methods:
        get_ch(ch_name), return channel by name.

    Note:
        Labber.LogFile holds logs from multiple experiments, while is not supported
        here. Entries from all logs are treat equal foot here.
    """
    def __init__(self, logfile):
        """Initialize a Logfile object from Labber.logfile.
        
        Args:
            logfile: Labber.logfile.
        """

        self.logfile = logfile
        self.step_channels = self.logfile.getStepChannels()  # Could be super time-costing!!
        self.log_channels = self.logfile.getLogChannels()  # type: List[dict]

    def __repr__(self):
        return f'Labber.Logfile("{self.path}")'

    @property
    def path(self):
        spath = self.logfile.getFilePath(tags=0)  # bug from Labber: 'tags' means nothing.
        return Path(spath).resolve()

    @property
    def ch_names(self):
        return [ch['name'] for ch in (self.step_channels + self.log_channels)]

    @property
    def num_of_entries(self):
        vec_log = [ch['name'] for ch in self.log_channels if ch['vector']]
        if vec_log:
            # If there is vector log channel, the max value for arguement of
            # getEntry() is given by getNumberOfEntries(name='chn_of_vector_log')
            # otherwise getNumberOfEntries() may return wrong value.
            return self.logfile.getNumberOfEntries(name=vec_log[0])
        else:
            return self.logfile.getNumberOfEntries()

    @property
    def user(self):
        return self.logfile.getUser()

    @user.setter
    def user(self, value):
        self.logfile.setUser(value)

    @property
    def tags(self):
        return self.logfile.getTags()

    @tags.setter
    def tags(self, value):
        self.logfile.setTags(value)

    @property
    def project(self):
        return self.logfile.getProject()

    @project.setter
    def project(self, value):
        return self.logfile.setProject(value)

    @property
    def comment(self):
        return self.logfile.getComment(log=-1)
    
    @comment.setter
    def comment(self, comment):
        self.logfile.setComment(comment, log=-1, set_all=True)

    def get_ch(self, ch_name):
        """Returns channel with given channel name.
        
        Args:
            ch_name: str.

        Returns:
            dict specifying channel settings.
        """
        for ch in (self.step_channels + self.log_channels):
            if ch['name'] == ch_name:
                return ch

class LabberRead(MyLogfile):
    """Open up a Labber logfile and read data from it.
    
    Attributes:
        dots: pandas.DataFrame,
            DataFrame with all datas from the logfile.
            Each row is a data point, Each column stores datas for one channels.
        ch_to_read: list(str),
            name of channels whose data will be returned in dots.

    MyLogfile:
    """
    __doc__ += MyLogfile.__doc__

    def __init__(
        self, 
        path: str, 
        ch_to_read=None,
        hold=False,
        log_chn=None, 
        step_chn=None,
        **kwargs,
    ):
        """Initialize a logfile with data (self.dots).
        
        Args:
            path: str, path to the logfile.
            hold: boolean, whether or not to read data to dots at initializing.
                Default is False.
            ch_to_read: 'all', List(str), name of channels to read, 
                overwriting step_chn and log_chn.
            log_chn: 'all' or List(str), name of log channels to read.
            step_chn: 'all' or List(str), name of step channels to read.
            kwargs: other arguments passed to get_dots().
        """

        path = Path(path).resolve()
        super().__init__(Labber.LogFile(path))

        if step_chn is None and kwargs.get('reserve_all_step'):
            print('Warning: arguments `reserve_all_step` is going to be removed!')
            kwargs.pop('reserve_all_step')
            step_chn = 'all'

        if ch_to_read:
            if ch_to_read == 'all':
                self.ch_to_read = self.get_ch_to_read(step_chn='all', log_chn='all')
            else:
                self.ch_to_read = ch_to_read
        else:
            self.ch_to_read = self.get_ch_to_read(log_chn=log_chn, step_chn=step_chn)

        # Read data into dots.
        if not hold:
            try:
                self.dots = self.get_dots(**kwargs)
            except Exception as exc:
                print(f'WARNING: Fail to load data. Error reports \n{exc}')
        else:
            self.dots = None  # A placeholder.

    @property
    def step_ch_read(self):
        """List of step channels whose value is read into dots."""
        return [ch for ch in self.step_channels 
            if ch['name'] in self.ch_to_read]

    def get_ch_to_read(self, log_chn=None, step_chn=None):
        """Get list of name of channels for get_entry or get_dots.
        
        Args:
            log_chn: 'all' or List(str), name of log channels to read.
            step_chn: 'all' or List(str), name of step channels to read.
        """

        if step_chn is None:
            step_chn = [ch['name'] for ch in self.step_channels 
                if len(ch['values']) > 1]  # Keep only those non-trivial channels.
        elif step_chn == 'all':
            step_chn = [ch['name'] for ch in self.step_channels]
        else:
            pass  # Keep it as given.

        if log_chn is None or log_chn == 'all':
            log_chn = [ch['name'] for ch in self.log_channels]
        else:
            pass  # Keep it as given.

        ch_to_read = step_chn + log_chn
        return ch_to_read

    def get_entry(self, index, ch_to_read=None, expand_vector=True):
        """Get data of a single entry from logfile, and return it in dots.
        
        Args:
            index: int, which entry to read.
            ch_to_read: list(str),
                name of channels whose data will be returned in dots.
                if None, use self.ch_to_read
            expand_vector: boolean,
                Whether to expand vector.
                Default is True.

        Returns:
            Dots with data from the entry.
        """
        if ch_to_read is None:
            ch_to_read = self.ch_to_read

        entry = self.logfile.getEntry(index)
        dot_dict = {
            k: v for k, v in entry.items()  # v can be scale, array or dict.
            if k in ch_to_read
        }

        # Expand vector trace into lists.
        # i.e. name: dict(y=[1,2,3], dt=1, t0=0) -> name: [1,2,3], name - index: [1,2,3]
        vec_chn = [k for k, v in dot_dict.items() if isinstance(v, dict)]
        for n in vec_chn:
            trace = dot_dict[n]
            dot_dict.update(
                {
                    n: trace['y'],
                    n+' - index': np.arange(
                        len(trace['y'])
                    ) * trace['dt'] + trace['t0']
                }
            )
        for trial in range(3):
            if expand_vector:
                # Expand element in vector traces on row of dots.
                # i.e. One vector trace -> many rows.
                try:
                    dots = pd.DataFrame(dot_dict)
                    break
                except ValueError as err:
                    if str(err) == 'arrays must all be same length':
                        # i.e. Vector log channels with different trace lengths
                        if trial == 2:  # The loop should be broken before trial reaches 2.
                            raise
                        else:
                            expand_vector = False
                            continue
                    else:
                        raise
            else:
                # Keep one vector trace in one row.
                dots = pd.DataFrame(columns=list(dot_dict.keys()))
                dots.loc[0] = list(dot_dict.values())  # FIXME: this may store complex data with dtype=object but not complex128.
                break
        
        dots['entry'] = index
        return dots.set_index('entry')


    def get_dots(self, index=None, ch_to_read=None, **kwargs):
        """Get all data from the logfile, return it in dots.

        Args:
            index: list(int),
                indice of entries to read.
                If None, read all entries.
            ch_to_read: list(str),
                name of channels whose data will be returned in dots.
                passed to self.get_entry()
            kwargs: other arguments passed to get_entry()

        Returns:
            Dots with data from specified entries.
        """

        if index is None:
            index = np.arange(self.num_of_entries)

        if self.num_of_entries > 1:
            list_entry = []
            for idx in tqdm(index, initial=0, total=self.num_of_entries):
                try:
                    list_entry.append(
                        self.get_entry(idx, **kwargs),
                    )
                except Exception as exc:
                    raise Exception(f'Problem happens in entry #{idx}') from exc
            dots = pd.concat(list_entry)
        else:
            dots = self.get_entry(0, **kwargs)
        return Dots(dots)

class LabberWrite(MyLogfile):
    """This class writes data to Labber logfile.
    
    Attributes:
        dots: pandas.DataFrame,
            Datas going to be saved to the logfile.

    Note:
        1. changs to self.dots will never save to the logfile unless explicitly calling
        self.save_dots().

    MyLogfile:
    """
    __doc__ += MyLogfile.__doc__

    def __init__(
        self,
        dots,
        path,
        user,  # a must-have metadata!
        log_channels=None,
        step_channels=None,
        cover_if_exist=False, 
        project=None,
        tags=None,
        comment=None,
    ):
        """Initialize a logfile with given data and metadatas.
        
        Args:
            dots: pd.DataFrame, datas to save.
            path: str, path to save the logfile.
                If points to a Labber database, files will be saved to the database.
            user: str.
            log_channels: None, list(str) or list(dict),
                Log channel should have: name, [unit], [x_name], [x_unit],
                x_name is needed only if the log channels comes with vector data,
                i.e. element of that column in dots is list.
                If None, all columns except step channels in dots will be taken 
                as log channels.
            step_channels: None, list(str) or list(dict),
                None means no explicit stepper.
                Step channel should have: name, [values], [unit], [combo_defs].
                Values will be completed according to dots.
            cover_if_exist: boolean,
                Cover if the given path already exists.
            project: str.
            tags: list(str).
            comment: str,
                metadatas for the logfile.

        Note:
            1. Specification of step_channels only affects GUI when Labber - Log 
            Viewer open the logfile & has no affect on saving datas in fact.
        """
        path = Path(path).resolve()
        # If path points to a database.
        database = parse_database(path)
        if database:
            # Update path with current time.
            now = datetime.now()
            path = (database 
                / f'{now:%Y}/{now:%m}/Data_{now:%m%d}'
                / path.name)

        if path.with_suffix('.hdf5').exists():
            if cover_if_exist:
                print(f'Warning: path {path} exists, covering it.')
                # Labber.createLogFile_ForData will cover it.
            else:
                raise FileExistsError(f'File exists at {path}\nand cover_if_exist is set False.')

        if database:
            if path.parent.exists():
                pass
            elif database.exists():
                path.parent.mkdir(parents=True)
            else:
                raise FileNotFoundError(f'No such database: {database}')

        entries, log_channels, step_channels = get_entries(dots, log_channels, 
            step_channels)

        logfile = Labber.createLogFile_ForData(path, log_channels, step_channels, 
            use_database=False)
        super().__init__(logfile)
            
        self.user = user
        self.project = project
        self.tags = tags
        self.comment = comment
        try:
            for i, entry in enumerate(tqdm(entries)):
                self.logfile.addEntry(entry)
        except Exception as exc:
            print(f'Exception happens at saving entry #{i}, reporting: {exc}\n'
                'Please try logfile.save_entries() again.')
        self.dots = dots
        self.entries = entries

    def save_entries(self):
        """Save self.entries into self.logfile with addEntry()."""
        try:
            for i, entry in enumerate(tqdm(self.entries)):
                self.logfile.addEntry(entry)
        except Exception as exc:
            raise Exception(f'Exception happens at saving entry #{i}') from exc

    def get_entries(self):
        return get_entries(self.dots, self.log_channels, self.step_channels)


def get_entries(dots, log_channels=None, step_channels=None):
    """Convert dots into entries for Labber.
    
    Args:
        dots: pandas.DataFrame, data to convert.
        log_channels: None, list(str) or list(dict),
            Log channel should have: name, [unit], [x_name], [x_unit],
            x_name is needed only if the log channels comes with vector data,
            i.e. element of that column in dots is list.
            If None, all columns except step channels in dots will be taken 
            as log channels.
        step_channels: None, list(str) or list(dict),
            None means no explicit stepper.
            Step channel should have: name, [values], [unit], [combo_defs].
            Values will be completed according to dots.

    Returns:
        entries: list[dict(quant=value)], the value can be scalar or 1d array.
        log_channels, step_channels, with completed contents.

    Note:
        1. The returned entries contains data from all columns in dots, no
        matter if it is specified in step_channels and log_chanenls or not,
        while Labber.addEntry() will bypass unnecessary items.
    """
    # Workout ch['name']
    if step_channels is None:
        step_channels = []
    elif isinstance(step_channels[0], str):
        step_channels = [dict(name=n) for n in step_channels]
    else:
        step_channels = deepcopy(step_channels)

    step_chn = [ch['name'] for ch in step_channels]

    if log_channels is None:
        if step_channels:
            log_channels = [dict(name=n) for n in dots.columns
                if n not in step_chn]
        else:
            log_channels = [dict(name=n) for n in dots.columns]
    elif isinstance(log_channels[0], str):
        log_channels = [dict(name=n) for n in log_channels]
    else:
        log_channels = deepcopy(log_channels)

    # Clear irrelavant columns in dots.
    log_chn = [ch['name'] for ch in log_channels]
    all_chn = [chn for chn in step_chn + log_chn if chn in dots.columns]
    dots = dots[all_chn]

    # Workout log_ch.complex, vector
    for ch in log_channels:
        col = dots[ch['name']]

        if col.dtype == np.dtype('object'):
            v = col.iloc[0]
            if isinstance(v, (list, tuple, np.ndarray)):
                ch['vector'] = True
                ch['complex'] = np.iscomplexobj(v)
            elif isinstance(v, complex):  # Sometimes complex data are stored with dtype=object in dots.
                ch['vector'] = False
                ch['complex'] = True
            else:
                raise TypeError(f'Invalid value type at column "{ch["name"]}"')
        else:
            ch['vector'] = False
            ch['complex'] = np.iscomplexobj(col)

    # Workout step_ch.values
    for ch in step_channels:
        if ch.get('values') is None:
            ch['values'] = dots[ch['name']].unique()

    # Workout entries
    exist_vector = np.any([ch['vector'] for ch in log_channels])
    if exist_vector:
        entries = [row.to_dict() for index, row in dots.iterrows()]  # TODO: Maybe dots.to_dict('record') is better.
        if len(step_channels) == 0:
            for ch in log_channels:
                ch['vector'] = True
        else:  # FIXME: Labber addEntry seems have trouble saving vector data along with non-vector data.
            for ch in log_channels:
                ch['vector'] = True
    else:
        if len(step_channels) == 0:
            entries = [dots.to_dict('list')]  # All data in one entry.
            for ch in log_channels:
                ch['vector'] = True
        elif len(step_channels) == 1:
            entries = [dots.to_dict('list')]  # All data in one entry.
            # Raise 'IndexError: list index out of range' if entry length > len(step_ch['values'])
        else:
            traces = dots.set_index(step_chn[1:][::-1])
            trace_index = traces.index.unique()
            entries = []
            entry_len = []
            for step_value in trace_index:
                trace = traces.loc[step_value, :]
                trace = trace.sort_values(by=[step_chn[0]])
                # Add trace x and y's.
                new_entry = trace.to_dict('list')
                # Add value of other steppers.
                if trace_index.nlevels > 1:
                    new_entry.update({key: val 
                        for key, val in zip(trace_index.names, step_value,)})
                else:
                    new_entry.update({trace_index.name: step_value})
                
                entries.append(new_entry)

                # Add entry length.
                if trace.ndim == 1:
                    entry_len.append(1)
                else:
                    entry_len.append(trace.shape[0])

            # Check length consistence of entries.
            if (np.max(entry_len)
                == np.min(entry_len)  # Uniform entry length.
                == len(step_channels[0]['values'])):
                pass
            else:  # If entries have different lengths.
                # Save the data in vector.
                new_log = step_channels.pop(0)
                del new_log['values']
                log_channels.append(new_log)
                for ch in log_channels:
                    ch['vector'] = True

    # Confirm values in entry are np.array with right shape.
    for entry in entries:
        entry_narray = {}
        for k, v in entry.items():
            v = np.array(v)  # Convert list into np.array.
            for ch in log_channels:
                if k == ch['name'] and ch['vector'] and v.ndim == 0:
                    v = np.array([v])  # ndim = 1 for vector data.
            entry_narray[k] = v
        entry.update(entry_narray)

    # Assign an non-empty unit for log_ch.
    for ch in log_channels:
        if ch.get('unit'):
            pass
        else:
            ch['unit'] = '_'

    # TODO: remove item in entries that are not in step_chn, log_chn or x_name.
    return entries, log_channels, step_channels

def get_ch_from_units_dict(units_dict):  # TODO: remove this.
    """Generate log channels list from given 'name: unit' pairs."""
    print('WARNING: This function (get_ch_from_units_dict) is going to be '
        'removed in future version.')
    log_channels = []
    for name, unit in units_dict.items():
        ch = {'name': name, 'unit': unit, 'complex': False, 'vector': False}
        log_channels.append(ch)
    return log_channels

def parse_database(path):
    """Path of the database where the path points to, if there is one.
    
    Returns: None or pathlib.Path
    """
    database = None
    if len(path.parts) >= 5:
        p_year = path.parts[-4]
        p_month = path.parts[-3]
        p_date = path.parts[-2]
        if p_date[5:7] == p_month:
            try:
                datetime.strptime(p_year, '%Y')
                datetime.strptime(p_date, 'Data_%m%d')
                database = path.parents[3]
            except Exception:
                pass
    return database


if __name__ == '__main__':
    pass
