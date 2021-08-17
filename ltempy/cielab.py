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

"""Generate rgba data from scalar values and implement the CIELAB colorspace.
"""

import numpy as np

__all__ = ['cielab_rgba']

def cielab_rgba(data, brightness = 'intensity', alpha = 'uniform'):
	"""Convert complex values to rgba values based on the CIELAB color space.

	Color represents the complex phase. Brightness and alpha may
	be uniform, or represent either amplitude or intensity.

	The CIELAB color space is designed to be perceptually uniform - none of the
	colors appear brighter or darker than the others.

	**Parameters**

	* **data** : _ndarray_ <br />
	An array with the data to convert. Dtype may be complex or real - if real,
	the color will be uniform, and values will be represented by brightness. Does not need to be normalized.

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
	shape MxN, the output array will have shape MxNx4.
	"""
	vals = {
		'uniform': lambda data: 255,
		'amplitude': lambda data: np.absolute(data) * 255,
		'intensity': lambda data: np.absolute(data)**2 * 255
		}
	if not brightness in vals.keys():
		raise ValueError("Invalid brightness value; should be one of {}".format(list(vals.keys())))
	if not alpha in vals.keys():
		raise ValueError("Invalid alpha value; should be one of {}".format(list(vals.keys())))

	data /= np.max(np.abs(data))
	rgba_image_components = np.zeros(np.append(data.shape, 4),dtype=np.uint8)
	bvalue = vals[brightness](data)
	avalue = vals[alpha](data)
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
