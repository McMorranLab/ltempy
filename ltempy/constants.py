# ltempy is a set of LTEM analysis and simulation tools developed by WSP as a member of the McMorran Lab
# Copyright (C) 2021  William S. Parker
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

r"""Allows unit scaling across `ltempy`.

By default, `ltempy` uses SI units. This module allows you to scale
the SI base units. So, for example, to work in kilometers rather than meters:

```python
import ltempy as lp
lp.set_units(meter = 1e-3) # set km as base unit for length

print(lp.constants.c) # yields 299792.458 km / s
```

Note that all other modules of `ltempy` and their functions update automatically as well.
So for example, if you set km as the base unit for length (`lp.set_units(meter = 1e-3)`),
and then run a sitie reconstruction with `defocus = 1`, you've just run sitie with defocus equal to 1km.

Also note that `ltempy` functions that use constants (such as `ind_from_phase`, which uses `constants.e` and
`constants.hbar`) will be affected, as both \(e\) (the electron charge) and \(\hbar\) (the reduced Planck constant),
have different numerical values when km is the base unit.
"""

__all__ = ['s','m','kg','A','K','mol','cd','c','hbar','e','pi','set_units']

def set_units(second=1, meter=1, kilogram=1, Ampere=1, Kelvin=1, mole=1, candela=1):
    """Sets the units across the ltempy module.

    For example, `set_units(meter = 1e3)` sets the millimeter as the base unit for length.

    **Parameters**

    * **second** : _number, optional_ <br />
    The SI base unit for time. <br />
    Default is `second = 1`.

    * **meter** : _number, optional_ <br />
    The SI base unit for length. <br />
    Default is `meter = 1`.

    * **kilogram** : _number, optional_ <br />
    The SI base unit for mass. <br />
    Default is `kilogram = 1`.

    * **Ampere** : _number, optional_ <br />
    The SI base unit for current. <br />
    Default is `Ampere = 1`.

    * **Kelvin** : _number, optional_ <br />
    The SI base unit for temperature. <br />
    Default is `Kelvin = 1`.

    * **mole** : _number, optional_ <br />
    The SI base unit for amount of substance. <br />
    Default is `mole = 1`.

    * **candela** : _number, optional_ <br />
    The SI base unit for luminous intensity. <br />
    Default is `candela = 1`.
    """
    global s,m,kg,A,K,mol,cd
    s,m,kg,A,K,mol,cd = second, meter, kilogram, Ampere, Kelvin, mole, candela
    set_consts(s, m, kg, A, K, mol, cd)

def set_consts(s, m, kg, A, K, mol, cd):
    """Utility function for `set_units()`."""
    global c, hbar, e, pi, mu0
    F = s**4 * A**2 / m**2 / kg
    J = kg * m**2 / s**2
    C = A * s
    W = kg * m**2 / s**3
    eV = 1.602176634e-19 * J

    c       = 299792458.0   			* m / s
    hbar    = 1.0545718176461565e-34	* J * s
    e       = 1.602176634e-19			* C
    pi      = 3.141592653589793
    mu0     = 1.25663706212e-6          * kg * m / s**2 / A**2

set_units()
