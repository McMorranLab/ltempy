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

"""Contains utilities for reconstructing phase and magnetization from Lorentz images.

The most common use case is to generate a Lorentz object from a `.dm3` or `.ser`file.
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
import matplotlib.pyplot as plt
import os

from . import image_processing as ip
from .pyplotwrapper import subplots
from . import constants as _
np.seterr(divide='ignore', invalid='ignore')
import json

__all__ = [
		'Lorentz',
		'lorentz',
		'ind_from_phase',
		'ind_from_img',
		'phase_from_img']

def lorentz(file):
	"""Creates a `Lorentz` class instance for each image in a `.dm3` sequence.

	This function acts as a wrapper for the `Lorentz` class, adding an extra handler
	for `.dm3` image sequences. When the `.dm3` file contains a sequence of images, it is split into
	individual `Lorentz` instances.

	**Parameters**

	* **dm3file** : _dictionary-like_ <br />
	a dm3-like file with the following keys: <br />
		<ul>
		<li> **data** : _ndarray_ <br />
		An array carrying the electron counts. </li>
		<li> **pixelSize** : _tuple_ <br />
		(_number_, _number_) - the x and y pixel sizes. </li>
		<li> **pixelUnit** : _tuple_ <br />
		(_string_, _string_) - the x and y pixel units. </li>
		<li> **filename** : _string_ <br /></li>
		</ul>

	**Returns**
	* **out** : _list, Lorentz_ <br />
	a list of `wsp-tools.sitie.Lorentz` instances, if the `.dm3` file is an image sequence.
	Otherwise, a single `wsp-tools.sitie.Lorentz` instance.
	"""
	if len(file['data'].shape) <= 2:
		return Lorentz(file)
	else:
		f1 = file.copy()
		out = []
		for dataset in file['data']:
			f1['data'] = dataset
			f1['pixelSize'] = [file['pixelSize'][-2], file['pixelSize'][-1]]
			f1['pixelUnit'] = [file['pixelUnit'][-2], file['pixelUnit'][-1]]
			f1['coords'] = [file['coords'][-2], file['coords'][-1]]
			out.append(Lorentz(f1))
		return(out)

class Lorentz:
	"""Class that contains sitie information about a lorentz image.

	**Parameters**

	* **dm3file** : _dictionary-like_ <br />
	a dm3-like file with the following keys: <br />
		<ul>
		<li> **data** : _ndarray_ <br />
		An array carrying the electron counts. </li>
		<li> **pixelSize** : _tuple_ <br />
		(_number_, _number_) - the x and y pixel sizes. </li>
		<li> **pixelUnit** : _tuple_ <br />
		(_string_, _string_) - the x and y pixel units. </li>
		<li> **filename** : _string_ <br /></li>
		</ul>
	"""
	def __init__(self, dm3file):
		self.data = ip.ndap(dm3file['data'])
		self.dx = dm3file['pixelSize'][0]
		self.dy = dm3file['pixelSize'][1]
		self.xUnit = dm3file['pixelUnit'][0]
		self.yUnit = dm3file['pixelUnit'][1]
		self.x = np.arange(0,self.data.shape[1]) * self.dx
		self.y = np.arange(0,self.data.shape[0]) * self.dy
		self.metadata = {
							'dx':float(dm3file['pixelSize'][0]),
							'dy':float(dm3file['pixelSize'][1]),
							'xUnit':dm3file['pixelUnit'][0],
							'yUnit':dm3file['pixelUnit'][1],
							'filename':dm3file['filename']
						}
		self.phase = None
		self.Bx, self.By = None, None
		self.fix_units()

	def fix_units(self, xunit=None, yunit=None):
		"""Change the pixel units to meters.

		**Parameters**

		* **unit** : _number, optional_ <br />
		The scale to multiply values by (i.e., going from 'µm' to 'm', you would use `unit = 1e-6`). If none is given, `fix_units` will try to convert from `self.pixelUnit` to meters.

		**Returns**

		* **self** : _lorentz_
		"""
		if xunit is None:
			if self.xUnit == 'nm':
				xunit = 1e-9
			elif self.xUnit == 'mm':
				xunit = 1e-3
			elif self.xUnit == 'µm':
				xunit = 1e-6
			elif self.xUnit == 'm':
				xunit = 1
			else:
				xunit = 1
		if yunit is None:
			if self.yUnit == 'nm':
				yunit = 1e-9
			elif self.yUnit == 'mm':
				yunit = 1e-3
			elif self.yUnit == 'µm':
				yunit = 1e-6
			elif self.yUnit == 'm':
				yunit = 1
			else:
				yunit = 1
		self.dx *= xunit
		self.dy *= yunit
		self.xUnit = 'm'
		self.yUnit = 'm'
		self.x *= xunit
		self.y *= yunit
		self.metadata.update({
			'dx': float(self.dx),
			'dy': float(self.dy),
			'xUnit': self.xUnit,
			'yUnit': self.yUnit
		})
		return(None)

	def sitie(self, df = 1, thickness = 60e-9, wavelength=1.97e-12):
		"""Carries out phase and B-field reconstruction.

		Assigns phase, Bx, and By attributes.

		Updates metadata with the defocus and wavelength.

		**Parameters**

		* **df** : _number, optional_ <br />
		The defocus at which the images were taken. <br />
		Default is `df = 1`.

		* **wavelength** : _number, optional_ <br />
		The electron wavelength. <br />
		Default is `wavelength = 1.96e-12` (relativistic wavelength of a 300kV electron).

		**Returns**

		* **self** : _lorentz_
		"""
		self.metadata.update({'defocus': df, 'wavelength': wavelength, 'thickness': thickness})
		self.phase = ip.ndap(phase_from_img(self.data, df, self.dx, self.dy, wavelength))
		self.Bx, self.By = [ip.ndap(arr) for arr in ind_from_phase(self.phase, thickness)]
		return(None)

	def preview(self, window=None):
		"""Preview the image.

		Note that unlike `pyplotwrapper`, window is in units of pixels.

		**Parameters**

		* **window** : _array-like, optional_ <br />
		Format is `window = (xmin, xmax, ymin, ymax)`. <br />
		Default is `window = (0, -1, 0, -1)`
		"""
		fig, ax = subplots(11)
		if not window is None:
			ax[0,0].setWindow(window)
		data = ip.clip_data(self.data, sigma=10)
		ax[0,0].setAxes(self.x, self.y)
		ax[0,0].set_xlabel("x ({:})".format(self.dx))
		ax[0,0].set_ylabel("y ({:})".format(self.dy))
		ax[0,0].imshow(data)
		plt.show()

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
	rhs = ip.shift_pos(rhs)
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
