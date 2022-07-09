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


"""Basic image processing tools.
"""

import os
from pathlib import Path
import numpy as np
from ._utils import _extend_and_fill_mirror
from warnings import warn

__all__ = ['high_pass', 'low_pass', 'gaussian_blur',
           'clip_data', 'shift_pos', 'outpath', 'ndap']


def high_pass(data, cutoff=1 / 1024, dx=1, dy=1, gaussian=False):
    """Apply a high-pass filter to a 2d-array.

    **Parameters**

    * **data** : _complex ndarray_ <br />

    * **cutoff** : _number, optional_ <br />
    Cutoff frequency of the tophat filter
    or standard deviation of the gaussian filter, measured in inverse units. 
    If `dx` and `dy` are 1 (default), the unit is pixels. <br />
    Default is `cutoff = 1 / 1024`.

    * **dx** : _number, optional_ <br />
    Pixel spacing. <br />
    Default is `dx = 1`. 

    * **dy** : _number, optional_ <br />
    Pixel spacing. <br />
    Default is `dy = 1`. 

    * **gaussian** : _boolean, optional_ <br />
    Note: this flag is being deprecated. For this functionality, use `scipy`'s butter function. 
    If true, a gaussian filter is used instead of a tophat. <br />
    Default is `gaussian = False`.

    **Returns**

    * _complex ndarray_ <br />
    """
    X = np.fft.fftfreq(data.shape[1], dx)
    Y = np.fft.fftfreq(data.shape[0], dy)
    x, y = np.meshgrid(X, Y)

    g = 1 - np.exp(-(x**2+y**2)/2/cutoff**2)
    if not gaussian:
        g[x**2 + y**2 > cutoff**2] = 1
        g[x**2 + y**2 <= cutoff**2] = 0
    else:
        warn(
            "Gaussian high-pass is being deprecated and will be removed in a future version of ltempy",
            DeprecationWarning,
            stacklevel=2
        )

    Fdata = np.fft.fft2(data)
    FFdata = np.fft.ifft2(g * Fdata)
    return(FFdata)


def low_pass(data, cutoff=1 / 4, dx=1, dy=1, gaussian=False):
    """Apply a low-pass filter to a 2d-array.

    **Parameters**

    * **data** : _complex ndarray_ <br />

    * **cutoff** : _number, optional_ <br />
    Cutoff frequency of the tophat filter
    or standard deviation of the gaussian filter, measured in inverse units. 
    If `dx` and `dy` are 1 (default), the unit is pixels. <br />
    Default is `cutoff = 1 / 4`.

    * **dx** : _number, optional_ <br />
    Pixel spacing. <br />
    Default is `dx = 1`. 

    * **dy** : _number, optional_ <br />
    Pixel spacing. <br />
    Default is `dy = 1`. 

    * **gaussian** : _boolean, optional_ <br />
    Note: this flag is being deprecated. For a gaussian low-pass, use `gaussian_blur`. 
    If true, a gaussian filter is used instead of a tophat. <br />
    Default is `gaussian = False`.

    **Returns**

    * _complex ndarray_ <br />
    """
    X = np.fft.fftfreq(data.shape[1], dx)
    Y = np.fft.fftfreq(data.shape[0], dy)
    x, y = np.meshgrid(X, Y)

    g = np.exp(-(x**2+y**2)/2/cutoff**2)
    if not gaussian:
        g[x**2 + y**2 < cutoff**2] = 1
        g[x**2 + y**2 >= cutoff**2] = 0
    else:
        warn(
            "Gaussian low-pass is being deprecated and will be removed in a future version of ltempy. Use `gaussian_blur` instead.",
            DeprecationWarning,
            stacklevel=2
        )

    Fdata = np.fft.fft2(data)
    FFdata = np.fft.ifft2(g * Fdata)
    return(FFdata)


def gaussian_blur(data, blur_radius=1):
    """Apply a Gaussian blur to the data. 

    **Parameters**

    * **data** : _complex ndarray_ <br />

    * **blur_radius** : _number, optional_ <br />
    The standard deviation of the Gaussian kernel, measured in pixels. <br />

    **Returns**

    * _complex ndarray_ <br />
    """
    ds0, ds1 = data.shape[0], data.shape[1]
    bdata = _extend_and_fill_mirror(data)
    X = np.fft.fftfreq(bdata.shape[1], 1)
    Y = np.fft.fftfreq(bdata.shape[0], 1)
    x, y = np.meshgrid(X, Y)

    g = np.exp(-2 * np.pi**2 * blur_radius**2 * (x**2 + y**2))

    Fdata = np.fft.fft2(bdata)
    FFdata = np.fft.ifft2(g * Fdata)
    return(FFdata[ds0:2*ds0, ds1:2*ds1])


def clip_data(data, sigma=5):
    """Clip data to a certain number of standard deviations from its average.

    **Parameters**

    * **data** : _complex ndarray_ <br />

    * **sigma** : _number, optional_ <br />
    Number of standard deviations from average to clip to. <br />
    Default is `sigma = 5`.

    **Returns**

    * _complex ndarray_ <br />
    """
    avg = np.mean(data)
    stdev = np.std(data)
    vmin = avg - sigma*stdev
    vmax = avg + sigma*stdev
    out = data.copy()
    out[out < vmin] = vmin
    out[out > vmax] = vmax
    return(out)


def shift_pos(data):
    """Shift data to be positive.

    **Parameters**

    * **data** : _ndarray_ <br />
    Dtype must be real.

    **Returns**

    * _ndarray_
    """
    return(data - np.min(data))


def outpath(datadir, outdir, fname, create=False):
    """A util to get the output filename.

    An example is easiest to explain:

    datadir: `/abs/path/to/data`

    fname: `/abs/path/to/data/plus/some/structure/too.dm3`

    outdir: `/where/i/want/to/write/data`

    This util can create the folder (if it doesn't exist):

    `/where/i/want/to/write/data/plus/some/structure`

    and return the filename:

    `/where/i/want/to/write/data/plus/some/structure/too`.

    **Parameters**

    * **datadir** : _string_ <br />
    The directory for the experiment's data.

    * **outdir** : _string_ <br />
    The main directory where you want outputs to go.

    * **fname** : _string_ <br />
    The name of the file in datadir.

    * **create** : _boolean, optional_ <br />
    Whether to create the output directory. <br />
    Default is `create = False`.

    **Returns**

    * _PurePath_ <br />
    The name of the file to save (without a suffix).
    """
    fname = os.path.splitext(fname)[0]
    subpath = os.path.relpath(os.path.dirname(fname), datadir)
    finoutdir = os.path.join(outdir, subpath)
    if create:
        if not os.path.exists(finoutdir):
            os.makedirs(finoutdir)
    return(Path(os.path.join(finoutdir, os.path.basename(fname))))


class ndap(np.ndarray):
    """A class that adds all the image processing methods to `np.ndarray`.

    The purpose of this class is just so you can write `myarray.high_pass().low_pass()`
    instead of `myarray = high_pass(low_pass(myarray))`.

    **Parameters**

    * **data** : _complex ndarray_ <br />
    Any type of ndarray - the methods are defined with a 2d array in mind.
    """
    def __new__(cls, data):
        dummy = np.asarray(data, dtype=data.dtype).copy().view(cls)
        return(dummy)

    def __init__(self, data, x=None, y=None):
        self.isComplex = np.iscomplexobj(data)

    def high_pass(self, cutoff=1 / 1024, dx=1, dy=1, gaussian=False):
        """Apply a high-pass filter to a 2d-array.

        **Parameters**

        * **cutoff** : _number, optional_ <br />
        Cutoff frequency of the tophat filter
        or standard deviation of the gaussian filter, measured in inverse units. 
        If `dx` and `dy` are 1 (default), the unit is pixels. <br />
        Default is `cutoff = 1 / 1024`.

        * **dx** : _number, optional_ <br />
        Pixel spacing. <br />
        Default is `dx = 1`. 

        * **dy** : _number, optional_ <br />
        Pixel spacing. <br />
        Default is `dy = 1`. 

        * **gaussian** : _boolean, optional_ <br />
        Note: this flag is being deprecated. For this functionality, use `scipy`'s butter function. 
        If true, a gaussian filter is used instead of a tophat. <br />
        Default is `gaussian = False`.

        **Returns**

        * _ndap_ <br />
        """
        if self.isComplex:
            self[:, :] = high_pass(self, cutoff, dx, dy, gaussian)
        else:
            self[:, :] = np.real(high_pass(self, cutoff, dx, dy, gaussian))
        return(self)

    def low_pass(self, cutoff=1 / 4, dx=1, dy=1, gaussian=False):
        """Apply a low-pass filter to a 2d-array.

        **Parameters**

        * **cutoff** : _number, optional_ <br />
        Cutoff frequency of the tophat filter
        or standard deviation of the gaussian filter, measured in inverse units. 
        If `dx` and `dy` are 1 (default), the unit is pixels. <br />
        Default is `cutoff = 1 / 4`.

        * **dx** : _number, optional_ <br />
        Pixel spacing. <br />
        Default is `dx = 1`. 

        * **dy** : _number, optional_ <br />
        Pixel spacing. <br />
        Default is `dy = 1`. 

        * **gaussian** : _boolean, optional_ <br />
        Note: this flag is being deprecated. For a gaussian low-pass, use `gaussian_blur`. 
        If true, a gaussian filter is used instead of a tophat. <br />
        Default is `gaussian = False`.

        **Returns**

        * _ndap_ <br />
        """
        if self.isComplex:
            self[:, :] = low_pass(self, cutoff, dx, dy, gaussian)
        else:
            self[:, :] = np.real(low_pass(self, cutoff, dx, dy, gaussian))
        return(self)

    def gaussian_blur(self, blur_radius=1):
        """Apply a Gaussian blur to the data. 

        **Parameters**

        * **data** : _complex ndarray_ <br />

        * **blur_radius** : _number, optional_ <br />
        The standard deviation of the Gaussian kernel, measured in pixels. <br />

        **Returns**

        * _ndap_ <br />
        """
        if self.isComplex:
            self[:, :] = gaussian_blur(self, blur_radius)
        else:
            self[:, :] = np.real(gaussian_blur(self, blur_radius))
        return(self)

    def clip_data(self, sigma=5):
        """Clip data to a certain number of standard deviations from its average.

        **Parameters**

        * **sigma** : _number, optional_ <br />
        Number of standard deviations from average to clip to. <br />
        Default is `sigma = 5`.

        **Returns**

        * _ndap_ <br />
        """
        self[:, :] = clip_data(self, sigma)
        return(self)

    def shift_pos(self):
        """Shift data to be positive.

        **Returns**

        * _ndap_ <br />
        """
        if self.isComplex:
            print("Positive shift not applied to complex data.")
            return(self)
        self[:, :] = shift_pos(self)
        return(self)
