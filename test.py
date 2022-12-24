# %%
from labcodes import fitter, models, fileio
# %%
from attrs import define, field

def f_out(cls, a):
    print(a)
    return cls

@define(slots=False)
class A:
    b = field(default=1)

    def f(self):
        return self.b

    @property
    def f_h(self):
        return f_out

    # def __attrs_post_init__(self):
    #     raise Exception('error')

a = A()
a.test = 1
a.f_h