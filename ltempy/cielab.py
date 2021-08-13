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

"""Tools to generate rgba data from scalar values and implement the CIELAB colorspace.
"""

import numpy as np
import matplotlib.pyplot as plt
from . import constants as _

__all__ = ['cielab_cmap','cielab_image','rgba']

def cielab_cmap(samples=256):
	"""Creates a `matplotlib.colors.ListedColormap` of the CIELAB color space.

	**Parameters**

	* **samples** : _number, optional_ <br />
	The number of samples. Any additional values will be nearest-neighbor interpolated, per matplotlib. <br />
	Default is `samples = 256`.

	**Returns**

	* **cmap** : _ListedColormap_ <br />
	A colormap, that can be used normally: `plt.imshow(data, cmap=cmap)`.
	"""
	from matplotlib.colors import ListedColormap
	angles = np.linspace(0,2*_.pi,samples)
	cvals = np.exp(1j*angles).reshape(1, samples)
	rgbavals = cielab_image(cvals).squeeze()/255
	cmap = ListedColormap(rgbavals)
	return(cmap)

def cielab_image(data, brightness = 'intensity', alpha = 'uniform'):
	"""Converts complex values to rgba data based on the CIELAB color space.

	The output color will represent the complex angle, and brightness may
	represent either intensity or amplitude.

	The CIELAB color space is intended to be perceptually uniform - none of the
	colors look brighter or darker than the others.

	**Parameters**

	* **data** : _ndarray_ <br />
	An array with the data to represent. Dtype may be complex or real - if real,
	the color will be uniform, and values will be represented by brightness.

	* **brightness** : _string, optional_ <br />
	Allowed values: `'intensity'`, `'amplitude'`, `'uniform'`. <br />
	Default is `brightness = 'intensity'`.

	* **alpha** : _string, optional_ <br />
	Allowed values: `'intensity'`, `'amplitude'`, `'uniform'`. Determines the alpha
	component of the rgba value. <br />
	Default is `alpha = 'uniform'`.

	**Returns**

	* **rgba_image_components** : _ndarray_ <br />
	The rgba components calculated from scalar values. If the input array has
	shape NxN, the output array will have shape NxNx4.
	"""
	data /= np.max(np.abs(data))
	rgba_image_components = np.zeros(np.append(data.shape, 4),dtype=np.uint8)
	if brightness == 'uniform':
		bvalue = 255
	elif brightness == 'intensity':
		bvalue = np.absolute(data)**2 * 255
	elif brightness == 'amplitude':
		bvalue = np.absolute(data) * 255
	if alpha == 'uniform':
		avalue = 255
	elif alpha == 'intensity':
		avalue = np.absolute(data)**2 * 255
	elif alpha == 'amplitude':
		avalue = np.absolute(data) * 255
	hue = (np.angle(data) + np.pi) / 2
	pi6 = np.pi/6
	def sin2(array, offset):
		return(np.sin(array-offset)**2)
	r = sin2(hue, .15*pi6) + 0.35 * sin2(hue, 3.15 * pi6)
	b = sin2(hue, 4.25 * pi6)
	g = .6*sin2(hue, 2*pi6) + 0.065 * sin2(hue, 5.05 * pi6) + 0.445*b + 0.33*r
	rgba_image_components[...,0] = (r * bvalue).astype(np.uint8)
	rgba_image_components[...,1] = (g * bvalue).astype(np.uint8)
	rgba_image_components[...,2] = (b * bvalue).astype(np.uint8)
	rgba_image_components[...,3] = np.full(data.shape, fill_value=avalue, dtype=np.uint8)
	return(rgba_image_components)

def rgba(mode, cmap = None, brightness = 'intensity', alpha = 'uniform'):
	"""Converts a 2d complex array to rgba data.

	**Parameters**

	* **mode** : _complex ndarray_ <br />
	An array with the data to represent. Dtype may be complex or real - if real,
	the color will be uniform, and values will be represented by brightness.

	* **cmap** : _string, optional_ <br />
	If `None`, the CIELAB color space will be used. Otherwise, any
	pyplot ScalarMappable may be used. <br />
	Default is `cmap = None`.

	* **brightness** : _string, optional_ <br />
	Allowed values: `'intensity'`, `'amplitude'`, `'uniform'`. <br />
	Default is `brightness = 'intensity'`.

	* **alpha** : _string, optional_ <br />
	Allowed values: `'intensity'`, `'amplitude'`, `'uniform'`. Determines the alpha
	component of the rgba value. <br />
	Default is `alpha = 'uniform'`.

	**Returns**

	* **rgba_image_components** : _ndarray_ <br />
	The rgba components calculated from scalar values. If the input array has
	shape NxN, the output array will have shape NxNx4.

	"""
	mode /= np.max(np.abs(mode))
	if cmap is None:
		out = cielab_image(mode, brightness, alpha)
		return(out)
	colormap = plt.cm.ScalarMappable(cmap=cmap)
	out = colormap.to_rgba(np.angle(mode))
	if alpha == 'intensity':
		out[...,-1] = np.abs(mode)**2
	elif alpha == 'amplitude':
		out[...,-1] = np.abs(mode)
	if brightness == 'intensity':
		out[...,0] *= np.abs(mode)**2
		out[...,1] *= np.abs(mode)**2
		out[...,2] *= np.abs(mode)**2
	elif brightness == 'amplitude':
		out[...,0] *= np.abs(mode)
		out[...,1] *= np.abs(mode)
		out[...,2] *= np.abs(mode)
	return(out)
