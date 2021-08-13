# ltempy is LTEM data analysis and simulation tools developed by WSP as a grad student in the McMorran Lab.
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

"""Allows unit scaling across the module.

To operate in different base units:

```python
import ltempy as lp
lp.set_units(meter = 1e-3) # set km as base unit for length

print(lp.constants.c) # outputs 299792.458
```

Note that all other modules update automatically as well.
"""

__all__ = ['s','m','kg','A','K','mol','cd','c','hbar','e','pi','set_units']

def set_units(second=1, meter=1, kilogram=1, Ampere=1, Kelvin=1, mole=1, candela=1):
	"""Sets the units across the ltempy module.

	i.e. set_units(meter = 1000) sets the millimeter as the base unit for length.

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
	global c, hbar, e, pi
	F = s**4 * A**2 / m**2 / kg
	J = kg * m**2 / s**2
	C = A * s
	W = kg * m**2 / s**3
	eV = 1.602176634e-19 * J

	c       = 299792458.0   * m / s
	hbar    = 1.0545718176461565e-34       * J * s
	e       = 1.602176634e-19        * C
	pi      = 3.141592653589793

set_units()
