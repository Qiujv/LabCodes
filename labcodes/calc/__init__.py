"""Package making numerical calculations of models more convenient.

Providing:
    - calculators (usually verified in previous experiments);
    - utilities for constructing new calculators.

Usage:
    foo = calc.Foo()
    foo.demo()

Feel free to copy and make your own calculators.
"""

import scipy.constants as const

from labcodes.calc.base import Calculator, dept
from labcodes.calc.qubits import (
    Capacitor,
    Junction,
    Transmon,
    RF_SQUID,
    Gmon,
    TCoupler,
    ThermalDistribution,
    Cable,
)
from labcodes.calc.resonator_qc import LC_CCoupled, LC_MCoupled
