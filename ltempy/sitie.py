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

The most common use case is to generate a Lorentz object from a `.dm3` or `.ser` file.
Then you can analyze using high_pass(), sitie(), clip_data(), etc.

Example:

```python
import ncempy.io.dm as dm
import ltempy as lp

fname = '/path/to/data.dm3'
dm3file = dm.dmReader(fname)

img = lp.lorentz(dm3file)
img.sitie(defocus=1e-3)
img.phase.clip_data(sigma=5).high_pass().low_pass()
img.Bx.clip_data(sigma=5).high_pass().low_pass()
img.By.clip_data(sigma=5).high_pass().low_pass()

### plot img.Bx, img.By, img.phase, img.data, img.rawData, etc
```
"""

# %%
import numpy as np
import os

from . import process
from . import constants as _
from ._utils import damping
np.seterr(divide='ignore', invalid='ignore')

__all__ = [
		'SITIEImage',
		'sitie_image',
		'ind_from_phase',
		'ind_from_img',
		'phase_from_img']

def sitie_image(datafile):
	"""Creates a `Lorentz` class instance for each image in a sequence.

	This function acts as a wrapper for the `Lorentz` class, adding an extra handler
	for image sequences. When the dictionary contains a sequence of images, it is split into
	individual `Lorentz` class instances.

	**Parameters**

	* **datafile** : _dictionary_ <br />
	a dictionary with the following keys: <br />
		<ul>
		<li> **data** : _ndarray_ <br />
		A 2d array of the electron counts, or an array of 2d arrays of the electron counts. </li>
		<li> **pixelSize** : _tuple_ <br />
		(_number_, _number_) - the x and y pixel sizes.
		For compatibility with `.dm3` files, the last 2 elements of the tuple are used. </li>
		<li> **pixelUnit** : _tuple_ <br />
		(_string_, _string_) - the x and y pixel units.
		For compatibility with `.dm3` files, the last 2 elements of the tuple are used. </li>
		</ul>

	**Returns**
	* **out** : _list, Lorentz_ <br />
	A list of `wsp-tools.sitie.Lorentz` instances, if the dictionary is an image sequence.
	Otherwise, a single `wsp-tools.sitie.Lorentz` instance.
	"""
	if len(datafile['data'].shape) == 2:
		return(SITIEImage(datafile))
	else:
		f1 = datafile.copy()
		out = []
		for dataset in datafile['data']:
			f1['data'] = dataset
			f1['pixelSize'] = [datafile['pixelSize'][-2], datafile['pixelSize'][-1]]
			f1['pixelUnit'] = [datafile['pixelUnit'][-2], datafile['pixelUnit'][-1]]
			out.append(SITIEImage(f1))
		return(out)

class SITIEImage:
	"""A LorentzImage object represents an image containing only magnetic contrast.

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
		"""Try to set the units to meters."""
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

		* **self** : _lorentz_
		"""
		if yr is None:
			yr = xr
		if y_unit is None:
			y_unit = x_unit
		self.x_unit = x_unit
		self.y_unit = y_unit
		self.dx *= xr
		self.dy *= yr
		self.x *= xr
		self.y *= yr
		return(self)

	def reconstruct(self, df = 1e-3, thickness = 60e-9, wavelength=1.97e-12):
		"""Carries out phase and B-field reconstruction.

		Assigns `self.phase`, `self.Bx`, and `self.By` attributes.

		**Parameters**

		* **df** : _number, optional_ <br />
		The defocus at which the images were taken. <br />
		Default is `df = 1e-3`.

		* **wavelength** : _number, optional_ <br />
		The electron wavelength. <br />
		Default is `wavelength = 1.96e-12` (relativistic wavelength of a 300kV electron).

		**Returns**

		* **self** : _lorentz_
		"""
		self.phase = process.ndap(phase_from_img(self.data, df, self.dx, self.dy, wavelength))
		self.Bx, self.By = [process.ndap(arr) for arr in ind_from_phase(self.phase, thickness)]
		return(self)

	def validate(self, dx, threshold = 0.9, damping = damping, **kwargs):
		QX = np.fft.fftfreq(len(self.x), dx)
		QY = np.fft.fftfreq(len(self.y), dx)
		qx, qy = np.meshgrid(QX, QY)
		test = np.exp(-damping(qx, qy, **kwargs))
		if not np.any(test > threshold):
			return("For these parameters, SITIE is valid only for features larger than your image size. ")
		min_feature_size = 1 / np.max(np.sqrt(qx**2 + qy**2)[test > threshold])
		return("For these parameters, SITIE is valid for feature sizes larger than {}m".format(min_feature_size))

def phase_from_img(img, defocus = 0, dx = 1, dy = 1, wavelength = 1.97e-12):
	"""Reconstruct the Aharonov-Bohm phase from a defocussed image.

	This is an implementation of the SITIE equation (Eq 10) from J. Chess et al., 2017, _Streamlined approach to mapping the magnetic induction of skyrmionic materials_.

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

	* **phase** : _ndarray_ <br />
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

def ind_from_img(img, defocus = 0, dx = 1, dy = 1, thickness = 60e-9, wavelength = 1.97e-12):
	"""Reconstruct the magnetic induction from a defocussed image.

	This is an implementation of the SITIE equation Eq (10) and Eq (11) from J. Chess et al., 2017, _Streamlined approach to mapping the magnetic induction of skyrmionic materials_.

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

	* **thickness** : _number, optional_ <br />
	The thickness of the sample. <br />
	Default is `thickness = 60e-9`.

	* **wavelength** : _number, optional_ <br />
	The relativistic electron wavelength. <br />
	Default is `wavelength = 1.97e-9`.

	**Returns**

	* **Bx** : _ndarray_ <br />
	A 2d array containing the reconstructed x-component of magnetic induction.

	* **By** : _ndarray_ <br />
	A 2d array containing the reconstructed y-component of magnetic induction.
	"""
	phase = phase_from_img(img, defocus, dx, dy, wavelength)
	Bx, By = ind_from_phase(phase, thickness)
	return(np.array([Bx.real, By.real]))

def ind_from_phase(phase, thickness = 60e-9):
	"""Calculate the magnetic induction given the Aharonov-Bohm phase shift.

	This is an implementation of Eq (11) from J. Chess et al., 2017, _Streamlined approach to mapping the magnetic induction of skyrmionic materials_.

	**Parameters**

	* **phase** : _ndarray_ <br />
	A 2d array containing the electron phase immediately after the sample.

	* **thickness** : _number, optional_ <br />
	The thickness of the sample. <br />
	Default is `thickness = 60e-9`.

	**Returns**

	* **Bx** : _ndarray_ <br />
	The x-component of the magnetic induction.

	* **By** : _ndarray_ <br />
	The y-component of the magnetic induction.
	"""
	dpdy, dpdx = np.gradient(phase)
	Bx = _.hbar/_.e/thickness * dpdy
	By = -_.hbar/_.e/thickness * dpdx
	return(np.array([Bx.real, By.real]))
