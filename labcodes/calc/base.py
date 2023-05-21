import inspect
from functools import wraps

import numpy as np


def dept(f):
    """Wrap class function to be a dependent on object attrbites.

    Try to find default value from object attributes for all arguments.
    """
    @wraps(f)
    def wrapped_f(self, *user_args, **user_kw):
        debug = user_kw.get('debug', False)
        final_kw = {}  # Function kwargs to be filled as required.
        fsig = inspect.signature(f)
        for i, arg_name in enumerate(fsig.parameters.keys()):
            # Search for required arguements one by one.
            if i+1 <= len(user_args):
                final_kw[arg_name] = user_args[i]
                continue
            if arg_name in user_kw:
                final_kw[arg_name] = user_kw[arg_name]
                continue
            if not hasattr(self, arg_name):
                continue  # Leave blank and proceed to and TypeError: missing argument in function call.

            attr = getattr(self, arg_name)
            if getattr(attr, 'is_dept', False):
                if attr == f: raise Exception('Loop evaluation!')  # TODO: Need a smarter checker.
                if debug is True: print(arg_name)
                final_kw[arg_name] = attr(**user_kw)  # Recursion here.
            else:
                final_kw[arg_name] = attr

        return f(self, **final_kw)
    wrapped_f.is_dept = True
    return wrapped_f


class Calculator(object):
    """Abstract class for calculators.
    
    calculators are classes who has two types of members in their attributes: 
    indept and dept. Indept are static quantities, while dept are dynamically 
    calculated according to indept. Dept are defined as class methods decorated 
    with '@dept'.

    This class defines some functions that helps quickly setup or create an
    snapshot for a calculator.

    Note: Controdicts with attrs.
    Note: Don't try to optimize the performance, it is not for that.
    """
    def __init__(self, **kwargs):  # TODO: __init__(self, a=2, b=3, ...)
        """Kwargs provided will be set to object attributes."""
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __getitem__(self, key):
        v = getattr(self, key)
        if getattr(v, 'is_dept', False):
            return v()
        else:
            return v

    def copy(self, **kwargs):
        """Returns a copy with updated attributes."""
        kw = self.__dict__.copy()
        kw.update(kwargs)
        return self.__class__(**kw)

    new = copy  # alias.

    def snapshot(self, scalar_only=True):
        d = {}
        for k in dir(self):
            if k.startswith('_'): continue
            if k == 'snapshot': continue  # Avoid infinite recursion.
            v = getattr(self, k, None)
            if isinstance(v, Calculator):
                d[k] = v.snapshot  # Recursively solve the indeps.
            elif getattr(v, 'is_dept', False):
                d[k] = v()
            elif callable(v):
                continue
            else:
                d[k] = v
            if scalar_only and not np.isscalar(d[k]):
                del d[k]
        return d
