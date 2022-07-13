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

"""Contains utilities for reconstructing phase and magnetization from Lorentz images.

This module implements the Single Image Transport of Intensity Equation (SITIE) [1].

The most common use case is to generate a `SITIEImage` object from a `.dm3` or `.ser` file,
then reconstruct phase and induction with `sitie()`.

Example:

```python
import ncempy.io.dm as dm
import ltempy as lp

fname = '/path/to/data.dm3'
dm3file = dm.dmReader(fname)

img = lp.lorentz(dm3file)
img.reconstruct(defocus=1e-3)

### plot img.Bx, img.By, img.phase, img.data, etc
```

---

1. Chess, J. J. et al. Streamlined approach to mapping the magnetic induction of skyrmionic materials. Ultramicroscopy 177, 78–83 (2017).

"""

# %%
import numpy as np
import os

from . import process
from . import constants as _
np.seterr(divide='ignore', invalid='ignore')

__all__ = [
		'SITIEImage',
		'ind_from_phase',
		'sitie']

class SITIEImage:
	"""A SITIEImage object represents a defocussed image containing only magnetic contrast.

	**Parameters**

	* **datafile** : _dictionary_ <br />
	a dictionary with the following keys: <br />
		<ul>
		<li> **data** : _ndarray_ <br />
		A 2d array of the electron counts. </li>
		<li> **pixelSize** : _tuple_ <br />
		(_number_, _number_) - the x and y pixel sizes. </li>
		<li> **pixelUnit** : _tuple_ <br />
		(_string_, _string_) - the x and y pixel units. <br />
		Allowed values are `"pm"`, `"nm"`, `"µm"`, `"um"`, `"mm"`, and `"m"`.</li>
		</ul>
	"""
	def __init__(self, datafile):
		self.data = process.ndap(datafile.get('data'))
		self.dx = datafile.get('pixelSize')[0]
		self.dy = datafile.get('pixelSize')[1]
		self.x_unit = datafile.get('pixelUnit')[0]
		self.y_unit = datafile.get('pixelUnit')[1]
		self.x = np.arange(0,self.data.shape[1]) * self.dx
		self.y = np.arange(0,self.data.shape[0]) * self.dy
		self.phase = None
		self.Bx, self.By = None, None
		self.fix_units()

	def fix_units(self):
		"""Set the units to meters."""
		if self.x_unit == 'm':
			xr = 1
		elif self.x_unit == 'mm':
			xr = 1e-3
		elif self.x_unit == 'µm' or self.x_unit == 'um':
			xr = 1e-6
		elif self.x_unit == 'nm':
			xr = 1e-9
		elif self.x_unit == 'pm':
			xr = 1e-12
		else:
			print("Failed to set units to meters - pixelUnit was not recognized.")
			return(self)
		if self.y_unit == 'm':
			yr = 1
		elif self.y_unit == 'mm':
			yr = 1e-3
		elif self.y_unit == 'µm' or self.y_unit == 'um':
			yr = 1e-6
		elif self.y_unit == 'nm':
			yr = 1e-9
		elif self.y_unit == 'pm':
			yr = 1e-12
		else:
			print("Failed to set units to meters - pixelUnit was not recognized.")
			return(self)
		return(self.set_units(xr, "m", yr, "m"))

	def set_units(self, xr=1, x_unit="", yr=None, y_unit=None):
		"""Change the pixel units.

		**Parameters**

		* **xr** : _number, optional_ <br />
		New x-units per old x-units. (For example, 1e3 to convert from meters to mm). <br />
		Default is `xr = 1`.

		* **x_unit** : _string, optional_ <br />
		The new x-units (e.g., "nm" or "µm"). <br />
		Default is `x_unit = ""`.

		* **yr** : _number, optional_ <br />
		New y-units per old y-units. (For example, 1e3 to convert from meters to mm).
		If left empty, defaults to `xr`<br />
		Default is `yr = None`.

		* **y_unit** : _string, optional_ <br />
		The new y-units (e.g., "nm" or "µm"). If left empty, defaults to `x_unit`. <br />
		Default is `y_unit = None`.

		**Returns**

		* _SITIEImage_
		"""
		if yr is None:
			yr = xr
		if y_unit is None:
			y_unit = x_unit
		self.x_unit = x_unit
		self.y_unit = y_unit
		self.dx = self.dx * xr
		self.dy = self.dy * yr
		self.x = self.x * xr
		self.y = self.y * yr
		return(self)

	def reconstruct(self, df = 1e-3, thickness = 60e-9, wavelength=1.97e-12):
		"""Carries out phase and B-field reconstruction.

		Assigns `self.phase`, `self.Bx`, and `self.By` attributes.

		**Parameters**

		* **df** : _number, optional_ <br />
		The defocus at which the images were taken. <br />
		Default is `df = 1e-3`.

		* **thickness** : _number, optional_ <br />
		The thickness of the sample. <br />
		Default is `thickness = 60e-9`.

		* **wavelength** : _number, optional_ <br />
		The electron wavelength. <br />
		Default is `wavelength = 1.96e-12` (relativistic wavelength of a 300kV electron).

		**Returns**

		* _SITIEImage_
		"""
		self.phase = process.ndap(sitie(self.data, df, self.dx, self.dy, wavelength))
		self.Bx, self.By = [process.ndap(arr) for arr in ind_from_phase(self.phase, dx = self.dx, dy = self.dy, thickness = thickness)]
		return(self)

	def validate(self, threshold = 0.1, divangle = 1e-5, defocus = 1):
		r"""Estimate the validity of the SITIE approximation for the given parameters.

		SITIE expands on the approximations of TIE by assuming coherent illumination (that is,
		low divergence angle relative to the defocus and the spatial frequencies to be resolved).
		Another way to say this is that the damping envelope \(g(q_{\perp})\) is negligible.

		This returns \(q_{\perp}\) satisfying

		\[\frac{(\pi \Theta_c \Delta f)^2}{2} q_{\perp}^2 = \text{threshold}\]

		where \(\Delta f\) is the defocus and \(\Theta_c\) is the divergence angle. This \(q_{\perp}\)
		is the largest spatial frequency for which the SITIE approximation is valid.

		**Parameters**

		* **threshold** : _number, optional_ <br />
		The largest value that you consider \(<< 1\). <br />
		Default is `threshold = 0.1`.

		* **divangle** : _number, optional_ <br />
		The divergence angle. <br />
		Default is `divangle = 1e-5`.

		* **defocus** : _number, optional_ <br />
		Default is `defocus = 1`.

		* ****kwargs** <br />
		Any arguments to be passed to the damping function.
		"""
		prefactor = np.abs((_.pi * divangle * defocus)**2 / 2)
		qp_max = np.sqrt(threshold / prefactor)
		return(qp_max)

def sitie(img, defocus = 0, dx = 1, dy = 1, wavelength = 1.97e-12):
	"""Reconstruct the Aharonov-Bohm phase from a defocussed image.

	This is an implementation of the SITIE equation from [1], Eqn (10).

	**Parameters**

	* **img** : _ndarray_ <br />
	The 2d image data.

	* **defocus** : _number, optional_ <br />
	Default is `defocus = 0`.

	* **dx** : _number, optional_ <br />
	The pixel spacing in the x-direction. <br />
	Default is `dx = 1`

	* **dy** : _number, optional_ <br />
	The pixel spacing in the y-direction. <br />
	Default is `dy = 1`.

	* **wavelength** : _number, optional_ <br />
	The relativistic electron wavelength. <br />
	Default is `wavelength = 1.97e-9`.

	**Returns**

	* _ndarray_ <br />
	A 2d array containing the reconstructed Aharonov-Bohm phase shift.
	"""
	Sx = np.fft.fftfreq(img.shape[1], dx)
	Sy = np.fft.fftfreq(img.shape[0], dy)
	sx, sy = np.meshgrid(Sx, Sy)
	rhs = np.nan_to_num(2 * _.pi / wavelength / defocus * (1 - img / np.mean(img)), posinf = 0, neginf = 0)
	rhs = process.shift_pos(rhs)
	rhs = np.fft.fft2(rhs) / -4 / _.pi**2 / (sx**2 + sy**2)
	rhs = np.nan_to_num(rhs, posinf = 0, neginf = 0)
	phase = np.fft.ifft2(rhs)
	return(phase.real)

def ind_from_phase(phase, dx = 1, dy = 1, thickness = 60e-9):
	r"""Calculate the magnetic induction given the Aharonov-Bohm phase shift.

	Calculated using

	\[\nabla_{\perp} \phi_m = -\frac{e\tau}{\hbar} \left[\mathbf{B}\times\hat{\mathbf{e}}_{z}\right]\]

	where \(\tau\) is the thickness, \(\mathbf{B}\) is the transverse magnetic field components, integrated over \(z\),
	and \(\hat{\mathbf{e}}_z\) is the direction of propagation.

	**Parameters**

	* **phase** : _ndarray_ <br />
	A 2d array containing the electron phase immediately after the sample.

	* **dx** : _number, optional_ <br />
	The pixel spacing in the x-direction. <br />
	Default is `dx = 1`

	* **dy** : _number, optional_ <br />
	The pixel spacing in the y-direction. <br />
	Default is `dy = 1`.

	* **thickness** : _number, optional_ <br />
	The thickness of the sample. <br />
	Default is `thickness = 60e-9`.

	**Returns**

	* **Bx** : _ndarray_ <br />
	The x-component of the magnetic induction.

	* **By** : _ndarray_ <br />
	The y-component of the magnetic induction.
	"""
	dpdx = np.gradient(phase, dy, axis=1)
	dpdy = np.gradient(phase, dx, axis=0)
	Bx = _.hbar/_.e/thickness * dpdy
	By = -_.hbar/_.e/thickness * dpdx
	return(np.array([Bx.real, By.real]))
