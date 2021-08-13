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

import os
from pathlib import Path
import numpy as np

__all__ = ['high_pass','low_pass','clip_data','shift_pos','outpath','ndap']

# %%
def high_pass(data, sigma = 7, tophat = False):
	"""Apply a high pass filter to a 2d-array.

	**Parameters**

	* **data** : _complex ndarray_ <br />

	* **sigma** : _number, optional_ <br />
	Standard deviation of the gaussian filter, or radius of the tophat filter, measured in pixels. <br />
	Default is `sigma = 7`.

	* **tophat** : _boolean, optional_ <br />
	If `True`, a tophat high pass is used instead of a gaussian. <br />
	Default is `tophat = False`.

	**Returns**

	* **FFdata** : _complex ndarray_ <br />
	"""
	X = np.fft.fftfreq(data.shape[1], 1/data.shape[1])
	Y = np.fft.fftfreq(data.shape[0], 1/data.shape[0])
	x, y = np.meshgrid(X, Y)

	g = 1 - np.exp(-(x**2+y**2)/2/sigma**2)
	if tophat:
		g[x**2 + y**2 > sigma**2] = 1
		g[x**2 + y**2 <= sigma**2] = 0

	Fdata = np.fft.fft2(data)
	FFdata = np.fft.ifft2(g * Fdata)
	return(FFdata)

def low_pass(data, sigma = 10, tophat = False):
	"""Apply a low pass filter to a 2d-array.

	**Parameters**

	* **data** : _complex ndarray_ <br />

	* **sigma** : _number, optional_ <br />
	Standard deviation of the gaussian filter, or radius of the tophat filter, measured in pixels. <br />
	Default is `sigma = 10`.

	* **tophat** : _boolean, optional_ <br />
	If `True`, a tophat low pass is used instead of a gaussian. <br />
	Default is `tophat = False`.

	**Returns**

	* **FFdata** : _complex ndarray_ <br />
	"""
	X = np.fft.fftfreq(data.shape[1], 1/data.shape[1])
	Y = np.fft.fftfreq(data.shape[0], 1/data.shape[0])
	x, y = np.meshgrid(X, Y)

	g = np.exp(-(x**2+y**2)/2/sigma**2)
	if tophat:
		g[x**2 + y**2 < sigma**2] = 1
		g[x**2 + y**2 >= sigma**2] = 0

	Fdata = np.fft.fft2(data)
	FFdata = np.fft.ifft2(g * Fdata)
	return(FFdata)

def clip_data(data, sigma = 5):
	"""Clip data to a certain number of standard deviations from average.

	* **data** : _complex ndarray_ <br />

	* **sigma** : _number, optional_ <br />
	Number of standard deviations from average to clip to. <br />
	Default is `sigma = 5`.

	**Returns**

	* **data** : _complex ndarray_ <br />
	"""
	avg = np.mean(data)
	stdev = np.std(data)
	vmin = avg - sigma*stdev
	vmax = avg + sigma*stdev
	data[data < vmin] = vmin
	data[data > vmax] = vmax
	return(data)

def shift_pos(data):
	"""Shift data to be all greater than zero.

	**Parameters**

	* **data** : _complex ndarray_ <br />

	**Returns**

	* **data** : _complex ndarray_
	"""
	return(data - np.min(data))

def outpath(datadir, outdir, fname):
	"""A util to get the output filename.

	An example is easiest to explain:

	datadir: `/abs/path/to/data`

	fname: `/abs/path/to/data/plus/some/structure/too.dm3`

	outdir: `/where/i/want/to/write/data`

	This util will create the folder (if not exists):

	`/where/i/want/to/write/data/plus/some/structure`

	and return the filename:

	`/where/i/want/to/write/data/plus/some/structure/too`.

	**Parameters**

	* **datadir** : _string_ <br />
	The directory for the experiment's data. (abspath)

	* **outdir** : _string_ <br />
	The main directory where you want outputs to go. (abspath)

	* **fname** : _string_ <br />
	The name of the file in datadir. (abspath)

	**Returns**

	* **outname** : _string_ <br />
	The name of the file to save. (abspath)
	"""
	if not os.path.exists(outdir):
		os.makedirs(outdir)
	fname = os.path.splitext(fname)[0]
	subpath = os.path.relpath(os.path.dirname(fname), datadir)
	finoutdir = os.path.join(outdir, subpath)
	if not os.path.exists(finoutdir):
		os.makedirs(finoutdir)
	return(Path(os.path.join(finoutdir, os.path.basename(fname))))

# %%
class ndap(np.ndarray):
	"""A class that adds all the image processing methods to np.ndarray.

	The purpose of this class is just so you can write `myarray.high_pass().low_pass()` instead of `myarray = high_pass(low_pass(myarray))`.

	**Parameters**

	* **data** : _complex ndarray_ <br />
	Any type of ndarray - the methods are defined with a 2d array in mind.
	"""
	def __new__(cls, data):
		dummy = np.asarray(data, dtype=data.dtype).copy().view(cls)
		return(dummy)

	def __init__(self, data):
		self.isComplex = np.iscomplexobj(data)

	def high_pass(self, sigma = 7, tophat = False):
		"""Apply a high pass filter to a 2d-array.

		**Parameters**

		* **sigma** : _number, optional_ <br />
		Standard deviation of the gaussian filter, measured in pixels. <br />
		Default is `sigma = 7`.

		**Returns**

		* **FFdata** : _ndap_ <br />
		"""
		if self.isComplex:
			self[:,:] = high_pass(self, sigma, tophat)
		else:
			self[:,:] = np.real(high_pass(self, sigma, tophat))
		return(self)

	def low_pass(self, sigma = 100, tophat = False):
		"""Apply a low pass filter to a 2d-array.

		**Parameters**

		* **sigma** : _number, optional_ <br />
		Standard deviation of the gaussian filter, measured in pixels. <br />
		Default is `sigma = 100`.

		**Returns**

		* **FFdata** : _ndap_ <br />
		"""
		if self.isComplex:
			self[:,:] = low_pass(self, sigma, tophat)
		else:
			self[:,:] = np.real(low_pass(self, sigma, tophat))
		return(self)

	def clip_data(self, sigma = 5):
		"""Clip data to a certain number of standard deviations from average.

		* **sigma** : _number, optional_ <br />
		Number of standard deviations from average to clip to. <br />
		Default is `sigma = 5`.

		**Returns**

		* **data** : _ndap_ <br />
		"""
		self[:,:] = clip_data(self, sigma)
		return(self)

	def shift_pos(self):
		"""Shift data to be all greater than zero.

		**Returns**

		* **data** : _ndap_
		"""
		self[:,:] = shift_pos(self)
		return(self)
