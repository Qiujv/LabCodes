# %%
from labcodes import fitter, models, fileio
# %%
from attrs import define, field

@define(slots=False)
class A:
    b = field(default=1)

    def f(self):
        return self.b

    def __attrs_post_init__(self):
        raise Exception('error')

a = A()