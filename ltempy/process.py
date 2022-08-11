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


def high_pass(data, cutoff=1 / 1024, dx=1, dy=1, padding=False):
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

    * **padding** : _boolean, optional_ <br />
    Whether to zero-pad the input. Will use mirror padding if True. <br />
    Default is `padding = True`.

    **Returns**

    * _complex ndarray_ <br />
    """
    ds0, ds1 = data.shape[0], data.shape[1]
    if padding:
        bdata = _extend_and_fill_mirror(data)
    else:
        bdata = data
    X = np.fft.fftfreq(bdata.shape[1], 1)
    Y = np.fft.fftfreq(bdata.shape[0], 1)
    x, y = np.meshgrid(X, Y)

    g = np.zeros_like(x)

    g[x**2 + y**2 > cutoff**2] = 1
    g[x**2 + y**2 <= cutoff**2] = 0
    Fdata = np.fft.fft2(bdata)
    FFdata = np.fft.ifft2(g * Fdata)
    if padding:
        return(FFdata[ds0:2*ds0, ds1:2*ds1])
    else:
        return(FFdata)

def low_pass(data, cutoff=1 / 4, dx=1, dy=1, padding=False):
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

    * **padding** : _boolean, optional_ <br />
    Whether to zero-pad the input. Will use mirror padding if True. <br />
    Default is `padding = True`.

    **Returns**

    * _complex ndarray_ <br />
    """
    ds0, ds1 = data.shape[0], data.shape[1]
    if padding:
        bdata = _extend_and_fill_mirror(data)
    else:
        bdata = data
    X = np.fft.fftfreq(bdata.shape[1], 1)
    Y = np.fft.fftfreq(bdata.shape[0], 1)
    x, y = np.meshgrid(X, Y)

    g = np.zeros_like(x)

    g[x**2 + y**2 < cutoff**2] = 1
    g[x**2 + y**2 >= cutoff**2] = 0
    Fdata = np.fft.fft2(bdata)
    FFdata = np.fft.ifft2(g * Fdata)
    if padding:
        return(FFdata[ds0:2*ds0, ds1:2*ds1])
    else:
        return(FFdata)

def gaussian_blur(data, blur_radius=1, padding = True):
    """Apply a Gaussian blur to the data. 

    **Parameters**

    * **data** : _complex ndarray_ <br />

    * **blur_radius** : _number, optional_ <br />
    The standard deviation of the Gaussian kernel, measured in pixels. <br />
    Default is `blur_radius = 1`.

    * **padding** : _boolean, optional_ <br />
    Whether to zero-pad the input. Will use mirror padding if True. <br />
    Default is `padding = True`.

    **Returns**

    * _complex ndarray_ <br />
    """
    ds0, ds1 = data.shape[0], data.shape[1]
    if padding:
        bdata = _extend_and_fill_mirror(data)
    else:
        bdata = data
    X = np.fft.fftfreq(bdata.shape[1], 1)
    Y = np.fft.fftfreq(bdata.shape[0], 1)
    x, y = np.meshgrid(X, Y)

    g = np.exp(-2 * np.pi**2 * blur_radius**2 * (x**2 + y**2))

    Fdata = np.fft.fft2(bdata)
    FFdata = np.fft.ifft2(g * Fdata)
    if padding:
        return(FFdata[ds0:2*ds0, ds1:2*ds1])
    else:
        return(FFdata)


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

    If `x` and `y` are given, then `dx` and `dy` are calculated and stored. 
    They will default to `dx = 1` and `dy = 1`, and will be used in `low_pass()`, 
    `high_pass()`, etc, unless specified otherwise in the function call. 

    **Parameters**

    * **data** : _complex ndarray_ <br />
    Any type of ndarray - the methods are defined with a 2d array in mind.

    * **x** : _ndarray_ <br />
    The x coordinates associated with the array data. 
    Must be a 1-dimensional ndarray with `len(x) == data.shape[1]`. 
    Note that `x` is associated with the second axis, to match the 
    convention of `meshgrid`.

    * **y** : _ndarray_ <br />
    The y coordinates associated with the array data. 
    Must be a 1-dimensional ndarray with 'len(y) == data.shape[0]`. 
    Note that `y` is associated with the first axis, to match the 
    convention of `meshgrid`.

    """
    ## __new__ and __array_finalize__ are directly from numpy docs
    ## their example is almost identical to this
    ## subclassing numpy arrays
    def __new__(cls, data, x=None, y=None):
        obj = np.asarray(data, dtype=data.dtype).copy().view(cls)
        if x is None:
          obj.x = np.arange(data.shape[1])
          obj.dx = obj.x[1] - obj.x[0]
        else:
          try: sel = obj.shape[1] != x.shape[0] or len(x.shape) != 1
          except: raise Exception("x must be a 1-dimensional numpy.ndarray. ")
          if sel:
            raise Exception("x shape must match the data's 1st axis. ")
          obj.x = x
          obj.dx = x[1] - x[0]
        if y is None:
          obj.y = np.arange(data.shape[0])
          obj.dy = obj.y[1] - obj.y[0]
        else:
          try: sel = obj.shape[0] != y.shape[0] or len(y.shape) != 1
          except: raise Exception("y must be a 1-dimensional numpy.ndarray. ")
          if sel:
            raise Exception("y shape must match the data's 1st axis. ")
          obj.y = y
          obj.dy = y[1] - y[0]
        obj.isComplex = np.iscomplexobj(data)
        return obj

    def __array_finalize__(self, obj):
      if obj is None: return
      self.x = getattr(obj, 'x', None)
      self.y = getattr(obj, 'y', None)
      self.dx = getattr(obj, 'dx', None)
      self.dy = getattr(obj, 'dy', None)
      self.isComplex = getattr(obj, 'isComplex', None)

    def high_pass(self, cutoff=1 / 1024, dx=None, dy=None, padding=False):
        """Apply a high-pass filter to a 2d-array.

        **Parameters**

        * **cutoff** : _number, optional_ <br />
        Cutoff frequency of the tophat filter
        or standard deviation of the gaussian filter, measured in inverse units. 
        If `dx` and `dy` are 1 (default), the unit is pixels. <br />
        Default is `cutoff = 1 / 1024`.

        * **dx** : _number, optional_ <br />
        Pixel spacing. <br />
        Default is `dx = self.dx`. 

        * **dy** : _number, optional_ <br />
        Pixel spacing. <br />
        Default is `dy = self.dy`. 

        * **padding** : _boolean, optional_ <br />
        Whether to zero-pad the input. Uses mirror padding if True. <br />
        Default is `padding = False`.

        **Returns**

        * _ndap_ <br />
        """
        if dx is None:
          dx = self.dx
        if dy is None:
          dy = self.dy
        if self.isComplex:
            self[:, :] = high_pass(self, cutoff, dx, dy, padding)
        else:
            self[:, :] = np.real(high_pass(self, cutoff, dx, dy, padding))
        return(self)

    def low_pass(self, cutoff=1 / 4, dx=1, dy=1, padding=False):
        """Apply a low-pass filter to a 2d-array.

        **Parameters**

        * **cutoff** : _number, optional_ <br />
        Cutoff frequency of the tophat filter
        or standard deviation of the gaussian filter, measured in inverse units. 
        If `dx` and `dy` are 1 (default), the unit is pixels. <br />
        Default is `cutoff = 1 / 4`.

        * **dx** : _number, optional_ <br />
        Pixel spacing. <br />
        Default is `dx = self.dx`. 

        * **dy** : _number, optional_ <br />
        Pixel spacing. <br />
        Default is `dy = self.dy`. 

        * **padding** : _boolean, optional_ <br />
        Whether to zero-pad the input. Uses mirror padding if True. <br />
        Default is `padding = False`.

        **Returns**

        * _ndap_ <br />
        """
        if dx is None:
          dx = self.dx
        if dy is None:
          dy = self.dy
        if self.isComplex:
            self[:, :] = low_pass(self, cutoff, dx, dy, padding)
        else:
            self[:, :] = np.real(low_pass(self, cutoff, dx, dy, padding))
        return(self)

    def gaussian_blur(self, blur_radius=1, padding=True):
        """Apply a Gaussian blur to the data. 

        **Parameters**

        * **data** : _complex ndarray_ <br />

        * **blur_radius** : _number, optional_ <br />
        The standard deviation of the Gaussian kernel, measured in pixels. <br />
        Default is `blur_radius = 1`.

        * **padding** : _boolean, optional_ <br />
        Whether to zero-pad the input. Uses mirror padding if True. <br />
        Default is `padding = False`.

        **Returns**

        * _ndap_ <br />
        """
        if self.isComplex:
            self[:, :] = gaussian_blur(self, blur_radius, padding)
        else:
            self[:, :] = np.real(gaussian_blur(self, blur_radius, padding))
        return(self)
      
    def get_window(self, window, step=1):
      """Get a windowed section of data. 

      **Parameters**

      * **window** : _tuple_ <br />
      [xmin, xmax, ymin, ymax]. Note that these are the x and y values, not their indices. 
      Also note that the default x and y attributes of the ndap object are just its indices. 

      **Returns**

      * _ndap_ <br />
      A 2d ndap with the windowed data and new axes.
      """
      argxmin = np.argmin(np.abs(self.x - window[0]))
      argxmax = np.argmin(np.abs(self.x - window[1]))
      argymin = np.argmin(np.abs(self.y - window[2]))
      argymax = np.argmin(np.abs(self.y - window[3]))
      xout = self.x[argxmin:argxmax:step].copy()
      yout = self.y[argymin:argymax:step].copy()
      dout = ndap(self[argymin:argymax:step, argxmin:argxmax:step].copy(), x=xout, y=yout)
      return(dout)

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
