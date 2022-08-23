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

The CIELAB colorspace is intended to be perceptually uniform. While it doesn't fully succeed,
it does well enough to use color and brightness as independent information channels.

In this module, the CIELAB colorspace is used as a representation of complex values,
where color represents phase and brightness represents magnitude.
"""

import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from . import constants as _

__all__ = ['cielab_rgba',
            'rgba',
            'cielab_cmap']

def cielab_cmap(samples=256):
    """Creates a `matplotlib.colors.ListedColormap` of the CIELAB color space.

    **Parameters**

    * **samples** : _integer, optional_ <br />
    The number of samples. Any additional values will be nearest-neighbor interpolated by matplotlib. <br />
    Default is `samples = 256`.

    **Returns**

    * _ListedColormap_ <br />
    A colormap. Many `matplotlib` functions that accept strings for the cmap argument
    also accept a `ListedColormap`: `plt.imshow(data, cmap=cmap)`.
    """
    angles = np.linspace(0,2*_.pi,samples)
    cvals = np.exp(1j*angles).reshape(1, samples)
    rgbavals = cielab_rgba(cvals).squeeze()/255
    cmap = ListedColormap(rgbavals)
    return(cmap)

def cielab_rgba(data, brightness = 'intensity', alpha = 'uniform', shift=0):
    """Convert complex values to rgba values based on the CIELAB color space.

    Color represents the complex phase, while brightness may represent amplitude,
    intensity (amplituded squared) or be uniform.

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

    * **shift** : _number, optional_ <br />
    Angle by which to shift the colors. Does not modify the angle of the data. <br />
    Default is `shift = 0`.  

    **Returns**

    * _ndarray_ <br />
    The calculated rgba components. If the input array has
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

    data = data / np.max(np.abs(data))
    rgba_image_components = np.zeros(np.append(data.shape, 4),dtype=np.uint8)
    bvalue = vals[brightness](data)
    avalue = vals[alpha](data)
    hue = (np.angle(data) + np.pi - shift) / 2
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

def rgba(mode, cmap = None, brightness = 'intensity', alpha = 'uniform', shift=0):
    """Converts a 2d complex array to rgba data.

    **Parameters**

    * **mode** : _complex ndarray_ <br />
    An array with the data to represent. Dtype may be complex or real.

    * **cmap** : _string, optional_ <br />
    A `matplotlib` ScalarMappable to use for colors. If left blank,
    the CIELAB color space will be used instead. <br />
    Default is `cmap = None`.

    * **brightness** : _string, optional_ <br />
    Allowed values: `'intensity'`, `'amplitude'`, `'uniform'`. <br />
    Default is `brightness = 'intensity'`.

    * **alpha** : _string, optional_ <br />
    Allowed values: `'intensity'`, `'amplitude'`, `'uniform'`. Determines the alpha (opacity)
    component of the rgba value. <br />
    Default is `alpha = 'uniform'`.

    * **shift** : _number, optional_ <br />
    Angle by which to shift the colors. Does not modify the angle of the data. <br />
    Default is `shift = 0`.  

    **Returns**

    * _ndarray_ <br />
    The calculated rgba components. If the input array has
    shape NxN, the output array will have shape NxNx4.
    """
    vals = {
        'uniform': lambda data: 255,
        'amplitude': lambda data: 255 * np.absolute(data),
        'intensity': lambda data: 255 * np.absolute(data)**2
        }
    if not brightness in vals.keys():
        raise ValueError("Invalid brightness value; should be one of {}".format(list(vals.keys())))
    if not alpha in vals.keys():
        raise ValueError("Invalid alpha value; should be one of {}".format(list(vals.keys())))
    if cmap is None:
        return(cielab_rgba(mode, brightness, alpha, shift))
    mode = mode / np.max(np.absolute(mode))
    colormap = plt.cm.ScalarMappable(cmap=cmap)
    out = colormap.to_rgba((np.angle(mode) - shift)%(2 * np.pi))
    out[...,-1] = vals[alpha](mode)
    out[...,0] = out[...,0] * vals[brightness](mode)
    out[...,1] = out[...,1] * vals[brightness](mode)
    out[...,2] = out[...,2] * vals[brightness](mode)
    return(out.astype(np.uint8))
