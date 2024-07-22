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

r"""Contains wrappers for `matplotlib.pyplot`, tailored to the presentation of LTEM data.

The primary feature is the `singleAx` object, which extends the `maplotlib.axes.Axes` object.
It adds methods for windowing data, quiver plots, CIELAB plots, and creating square insets and a colorwheel.

The typical use case is to use `subplots` to generate a figure and axes:

```python
import numpy
import ltempy
# Generate data
X = numpy.linspace(-1, 1, 128)
Y = numpy.linspace(-3, 3, 3 * 128)
x, y = numpy.meshgrid(X, Y)
z = x + 1j*y
f = numpy.sin(z)

# Plot data
window = (.3, .7, .3, .7)
fig, [[ax1, ax2]] = ltempy.subplots(12)
ax1.set_axes(X, Y)
ax1.inset(window)
ax1.imshow(numpy.abs(f)**2)
ax2.set_axes(ax1.x, ax1.y)
ax2.set_window(window)
ax2.cielab(f)
ax2.quiver(f, step=4)
plt.show()
```
"""

import numpy as np
from .colors import cielab_rgba, rgba, cielab_cmap
from . import constants as _

import matplotlib.pyplot as plt


__all__ = ['singleAx', 'subplots']

class singleAx():
    """An extension of the `matplotlib.axes.Axes` class.

    This class adds macros for 2d plotting. In particular,
    it's easy to select only a window of your data to show, to add x-y axes,
    to add an inset, to show the rgba version of a complex 2d array, and to
    show a quiver plot of a complex 2d array.

    Typical usage:

    ```python
    X = np.linspace(-10,10,xres)
    Y = np.linspace(-10,10,yres)
    x, y = np.meshgrid(X, Y)
    data = x+1j*y
    window = [-3,7,1,4]

    fig, ax = plt.subplots()
    myax = ltempy.singleAx(ax)
    myax.set_axes(x, y)
    myax.set_window(window)
    myax.set_xytitle('x','y','title')
    myax.cielab(data)
    plt.show()
    ```

    You also have direct access to the `matplotlib.pyplot.axes.Axes` object via `myax.ax`.

    More commonly, this class is returned by `ltempy.plot.subplots`.

    **Parameters**

    * **ax** : _matplotlib.axes.Axes_ <br />

    **Returns**

    * _ltempy.plot.singleAx_ <br />
    """
    def __init__(self, ax):
        self.ax = ax
        self.origin = 'lower'
        self.shift = 0
        self.x = None
        self.y = None
        self.xmin = None
        self.ymin = None
        self.xmax = None
        self.ymax = None

    def set_title(self, title='', **kwargs):
        """Sets the title of the plot.

        **Parameters**

        * **title** : _string, optional_ <br />
        The plot title. <br />
        Default is `title = ""`.

        * ****kwargs** <br />
        All other kwargs are passed on to `matplotlib.axes.Axes.set_title`.

        **Returns**

        * _singleAx_
        """
        self.ax.set_title(title, **kwargs)
        return(self)

    def set_xlabel(self, xlabel='', **kwargs):
        """Sets the xlabel of the plot.

        **Parameters*

        * **xlabel** : _string, optional_ <br />
        The xlabel. <br />
        Default is `xlabel = ""`.

        * ****kwargs** <br />
        All other kwargs are passed on to `matplotlib.axes.Axes.set_xlabel`.

        **Returns**

        * _singleAx_
        """
        self.ax.set_xlabel(xlabel, **kwargs)
        return(self)

    def set_ylabel(self, ylabel='', **kwargs):
        """Sets the ylabel of the plot.

        **Parameters**

        * **ylabel** : _string, optional_ <br />
        The ylabel. <br />
        Default is `ylabel = ""`.

        * ****kwargs** <br />
        All other kwargs are passed on to `matplotlib.axes.Axes.set_ylabel`.

        **Returns**

        * _singleAx_
        """
        self.ax.set_ylabel(ylabel, **kwargs)
        return(self)

    def set_xytitle(self, xlabel='', ylabel='', title='', **kwargs):
        """Set the xlabel, ylabel, and title at the same time.

        Sets all three even if not all are given.

        For individual control, use `singleAx.set_xlabel`, `singleAx.set_ylabel`,
        or `singleAx.set_title`.

        **Parameters**

        * **ylabel** : _string, optional_ <br />
        The ylabel. <br />
        Default is `ylabel = ""`.

        * **xlabel** : _string, optional_ <br />
        The xlabel. <br />
        Default is `xlabel = ""`.

        * **title** : _string, optional_ <br />
        The plot title. <br />
        Default is `title = ""`.

        * ****kwargs** <br />
        All other kwargs are passed on
        to `matplotlib.axes.Axes.set_xlabel`, `matplotlib.axes.Axes.set_ylabel`,
        and `matplotlib.axes.Axes.set_title`.

        **Returns**

        * _singleAx_
        """
        self.ax.set_xlabel(xlabel, **kwargs)
        self.ax.set_ylabel(ylabel, **kwargs)
        self.ax.set_title(title, **kwargs)
        return(self)

    def set_axes(self, x, y):
        """Sets the x and y axes of the singleAx object.

        Note that this can be used before or after `set_window()`,
        but make sure the two are in the same units.

        **Parameters**

        * **x** : _ndarray_ <br />
        The x-coordinates. Should be 1-dimensional.

        * **y** : _ndarray_ <br />
        The y-coordinates. Should be 1-dimensional.

        **Returns**

        * _singleAx_
        """
        self.x = x
        self.y = y
        return(self)

    def set_window(self, window):
        """Applies a window to the 'singleAx' object.

        Note that this can be used before or after `set_axes()`,
        but make sure the two are in the same units.

        **Parameters**

        * **window** : _array-like_ <br />
        Format: `window = [xmin, xmax, ymin, ymax]`. Note that these are the x
        and y values, rather than their indices.

        **Returns**

        * _singleAx_
        """
        self.xmin = window[0]
        self.xmax = window[1]
        self.ymin = window[2]
        self.ymax = window[3]
        return(self)

    def _pre_plot(self, data, step=1):
        if self.x is None:
                self.x = np.arange(0, data.shape[1])
        #   self.x = np.linspace(0, 100, data.shape[1])
        if self.y is None:
                self.y = np.arange(0, data.shape[0])
        #   self.y = np.linspace(0, 100, data.shape[0])
        self.dx = self.x[1]-self.x[0]
        self.dy = self.y[1]-self.y[0]
        if self.xmin is None:
            self.xmin = self.x[0]
        if self.xmax is None:
            self.xmax = self.x[-1]
        if self.ymin is None:
            self.ymin = self.y[0]
        if self.ymax is None:
            self.ymax = self.y[-1]
        argxmin = np.argmin(np.abs(self.x - self.xmin))
        argxmax = np.argmin(np.abs(self.x - self.xmax))+1
        argymin = np.argmin(np.abs(self.y - self.ymin))
        argymax = np.argmin(np.abs(self.y - self.ymax))+1
        dout = data[argymin:argymax:step, argxmin:argxmax:step]
        xout = self.x[argxmin:argxmax:step]
        yout = self.y[argymin:argymax:step]
        return(xout, yout, dout)

    def imshow(self, data, step=1, colorbar=False, **kwargs):
        """Imshows the (windowed) data.

        **Parameters**

        * **data** : _ndarray_ <br />
        The data to be shown. Use the un-windowed data - the window will be
        applied automatically, if you have set one.

        * **step** : _int, optional_ <br />
        data will be shown as `data[::step,::step]`. <br />
        Default is `step = 1`.

        * **colorbar** : _boolean, optional_ <br />
        If `True`, a colorbar will be added next to the plot. For more control over its
        appearance, see `singleAx.colorbar()`. <br />
        Default is `colorbar = False`.

        * ****kwargs** <br />
        All other kwargs are passed on to `matplotlib.axes.Axes.imshow`.

        **Returns**

        * _matplotlib.AxesImage_ <br />
        Returns the 'AxesImage' returned by `matplotlib.axes.Axes.imshow()`.
        """
        imshowargs = {'origin': self.origin}
        imshowargs.update(kwargs)
        x, y, d = self._pre_plot(data, step)
        if imshowargs['origin'] == 'lower':
            extent = [x[0]-self.dx/2, x[-1]+self.dx/2, y[0]-self.dy/2, y[-1]+self.dy/2]
        elif imshowargs['origin'] == 'upper':
            extent = [x[0]-self.dx/2, x[-1]+self.dx/2, y[-1]+self.dy/2, y[0]-self.dy/2]
        imshowargs.update({'extent': extent})
        imshowargs.update(kwargs)
        im = self.ax.imshow(d, **imshowargs)
        if colorbar:
            self.colorbar(im)
        return(im)

    def quiver(self, data, step=1, origin=None, **kwargs):
        """Shows a quiver plot of complex data.

        **Parameters**

        * **data** : _ndarray, complex_ <br />
        The data to be shown. Real part is x-component, imaginary is y-component.
        Use the un-windowed data - the window will be applied automatically, if you set one.

        * **step** : _int_ <br />
        data will be shown as `data[::step,::step]`. <br />
        Default is `step = 1`.

        * **origin** : _string_ <br />
        Either 'upper' or 'lower'. <br />
        Default is `self.origin`, whose default is 'lower'.

        * ****kwargs** <br />
        All other kwargs are passed on to `matplotlib.axes.Axes.quiver`.

        **Returns**

        * _matplotlib.quiver.Quiver_ <br />
        Returns the `Quiver` object returned by `matplotlib.axes.Axes.quiver()`.
        """
        if origin is None:
            origin = self.origin
        x, y, d = self._pre_plot(data, step)
        d = d.astype(complex)
        if origin == 'upper':
            d.imag = -1 * d.imag
        return(self.ax.quiver(x, y, d.real, d.imag, **kwargs))

    def cielab(self, data, step=1, brightness='intensity', alpha='uniform', **kwargs):
        """Show a CIELAB interpretation of complex data.

        Color represents phase, while brightness may be uniform or represent amplitude or intensity.

        **Parameters**

        * **data** : _complex ndarray_ <br />
        An array with the data to represent. Dtype may be complex or real.

        * **step** : _int_ <br />
        data will be shown as `data[::step,::step]`. <br />
        Default is `step = 1`.

        * **brightness** : _string, optional_ <br />
        Allowed values: `'intensity'`, `'amplitude'`, `'uniform'`. <br />
        Default is `brightness = 'intensity'`.

        * **alpha** : _string, optional_ <br />
        Allowed values: `'intensity'`, `'amplitude'`, `'uniform'`. Determines the alpha
        component of the rgba value. <br />
        Default is `alpha = 'uniform'`.

        * ****kwargs** <br />
        All other kwargs are passed on to `matplotlib.axes.Axes.imshow`.

        **Returns**

        * _matplotlib.AxesImage_ <br />
        Returns the 'AxesImage' returned by `matplotlib.axes.Axes.imshow()`.
        """
        imshowargs = {'origin': self.origin}
        imshowargs.update(kwargs)
        x, y, d = self._pre_plot(data, step)
        d = d.astype(complex)
        if imshowargs['origin'] == 'lower':
        #   extent = [x[0], x[-1], y[0], y[-1]]
            extent = [x[0]-self.dx/2, x[-1]+self.dx/2, y[0]-self.dy/2, y[-1]+self.dy/2]
        elif imshowargs['origin'] == 'upper':
            extent = [x[0]-self.dx/2, x[-1]+self.dx/2, y[-1]+self.dy/2, y[0]-self.dy/2]
        #   extent = [x[0], x[-1], y[-1], y[0]]
            d.imag = -1 * d.imag
        imshowargs.update({'extent': extent})
        imshowargs.update(kwargs)
        im = self.ax.imshow(rgba(d, brightness=brightness, alpha=alpha, shift=self.shift), **imshowargs)
        return(im)

    def colorbar(self, axesImage, position='right', size='5%', pad=0.05, **kwargs):
        """Append a colorbar to the `axes` object, based on the provided `axesImage`.

        There are two ways to apply a colorbar. You can provided `colorbar = True` to `ax.imshow()`. This will create a colorbar with some default parameters.
        Otherwise, if you want to customize the colorbar, you can use this method to do the following:
        ```python
        fig, [[ax]] = ltempy.subplots()
        im = ax.imshow(data)
        ax.colorbar(im)
        plt.show()
        ```
        and pass extra arguments to `colorbar()`.

        **Parameters**

        * **axesImage** : _matplotlib.image.AxesImage_ <br />
        This is what is returned by, for example, `plt.imshow()`.

        * **position** : _string_ <br />
        Accepted values are `'left'`, `'right'`, `'top'`, `'bottom'`. <br />
        Default is `position = "right"`.

        * **size** : _string_ <br />
        Must be `mpl_toolkits.axes_grid.axes_size` compatible. For example, give a percent. <br />
        Default is `size = "5%"`.

        * **pad** : _number_ <br />
        Default is `pad = 0.05`.

        * ****kwargs** <br />
        All further kwargs are passed to `Figure.colorbar()`.

        **Returns**

        * _matplotlib.colorbar.colorbar_
        """
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(self.ax)
        cax = divider.append_axes(position, size, pad)
        fig = self.ax.get_figure()
        cbar = fig.colorbar(axesImage, cax=cax, **kwargs)
        return(cbar)

    def inset(self, window, **kwargs):
        """Plots a square box with vertices defined by window.

        Default color is white.

        **Parameters**

        * **window** : _array-like_ <br />
        Format: `window = [xmin, xmax, ymin, ymax]`. Note that these are the x
        and y values, rather than their indices.

        * ****kwargs** <br />
        All other kwargs are passed on to `matplotlib.axes.Axes.plot`.

        **Returns**

        * _singleAx_
        """
        plotargs = {'color': 'white', 'linewidth': .5}
        plotargs.update(kwargs)
        self.ax.plot(np.linspace(window[0], window[1], 100),
                        np.zeros(100) + window[2], **plotargs)
        self.ax.plot(np.linspace(window[0], window[1],100),
                        np.zeros(100)+window[3], **plotargs)
        self.ax.plot(np.zeros(100) + window[0],
                        np.linspace(window[2], window[3], 100), **plotargs)
        self.ax.plot(np.zeros(100) + window[1],
                        np.linspace(window[2], window[3], 100),
                        **plotargs)
        return(self)

    def colorwheel(self, res=128, scale=0.25, cmap=None, 
                   brightness='intensity', alpha='uniform', 
                   align_x='right', align_y='bottom',
                   **kwargs):
        """Adds a colorwheel to the bottom right corner of the plot.

        **Parameters**

        * **res** : _int_ <br />
        The resolution of the colorwheel. <br />
        Default is `res = 128`.

        * **scale** : _float_ <br />
        The size of the colorwheel, in units of the width of the axis. <br />
        Default is `scale = 0.25`.

        * **cmap** : _string, optional_ <br />
        If `cmap = None`, the CIELAB color space will be used. Otherwise, any
        pyplot ScalarMappable may be used. <br />
        Default is `cmap = None`.

        * **brightness** : _string, optional_ <br />
        Allowed values: `'intensity'`, `'amplitude'`, `'uniform'`. <br />
        Default is `brightness = 'intensity'`.

        * **alpha** : _string, optional_ <br />
        Allowed values: `'intensity'`, `'amplitude'`, `'uniform'`. Determines the alpha
        component of the rgba value. <br />
        Default is `alpha = 'uniform'`.

        * ****kwargs** <br />
        All other kwargs are passed on to `matplotlib.axes.Axes.imshow`.

        **Returns**

        * _singleAx_
        """
        imshowargs = {'origin': self.origin, 'zorder': 3}
        imshowargs.update(kwargs)
        X = np.linspace(-1, 1, res)
        x, y = np.meshgrid(X, X)
        z = x + 1j*y
        sel = np.abs(z) > 1
        z[sel] = 0
        colors = rgba(z, brightness=brightness, alpha=alpha, cmap=cmap, shift=self.shift)
        colors[:,:,0][sel] = 0
        colors[:,:,1][sel] = 0
        colors[:,:,2][sel] = 0
        extent = self._get_colorwheel_extent(scale=scale, align_x=align_x, align_y=align_y)
        imshowargs.update(kwargs)
        self.ax.imshow(colors, extent=extent, **imshowargs)
        return(self)

    def _get_colorwheel_extent(self, scale=0.25, align_x='left', align_y='lower'):
        allowed_x_values = ['left', 'right']
        if align_x not in allowed_x_values:
            raise ValueError(f"align_x must be one of: {allowed_x_values}. ")
        allowed_y_values = ['top', 'bottom', 'upper', 'lower']
        if align_y not in allowed_y_values:
            raise ValueError(f"align_y must be one of {allowed_y_values}.")
        xlims = self.ax.get_xlim()
        xrange = xlims[1] - xlims[0]
        ylims = self.ax.get_ylim()
        yrange = ylims[1] - ylims[0]
        # Not sure why these next two lines are needed, but they are
        self.ax.set_xlim(xlims)
        self.ax.set_ylim(ylims)
        self.ax.set_aspect('equal')

        if align_x == 'left':
            xmin = xlims[0]
            xmax = xlims[0] + scale * xrange
        elif align_x == 'right': 
            xmin = xlims[1] - scale * xrange
            xmax = xlims[1]
        if align_y in ['top', 'upper']:
            if self.origin == 'lower':
                ymin = ylims[1] - scale * xrange
            elif self.origin == 'upper':
                ymin = ylims[1] + scale * xrange
            ymax = ylims[1]
        elif align_y in ['bottom', 'lower']:
            ymin = ylims[0]
            if self.origin == "lower":
                ymax = ylims[0] + scale * xrange
            elif self.origin == "upper":
                ymax = ylims[0] - scale * xrange
        if self.origin == 'lower':
            return [xmin, xmax, ymin, ymax]
        elif self.origin == 'upper':
            return [xmin, xmax, ymax, ymin]
        else:
            raise ValueError("Axis origin must be one of: ['lower', 'upper']. ")


def subplots(rc=11, **kwargs):
    """Creates a figure and axes, where each axis is a `singleAx` instance.

    Behaves like `matplotlib.pyplot.subplots()`, but replaces each
    `matplotlib.axes.Axes` object with a `wsp_tools.plot.singleAx`
    object.

    Each `wsp_tools.plot.singleAx` object in turn behaves like a
    normal `Axes` object, but with added methods.

    Note: two default kwargs are set. `'tight_layout': True` and `'squeeze': False`.

    **Parameters**

    * **rc** : _int_ <br />
    First digit - nrows. Second digit - ncols. <br />
    Default is `rc = 11`.

    * ****kwargs** <br />
    All other kwargs are passed on to `matplotlib.axes.Axes.subplots`.

    **Returns**

    * _Figure_ <br />

    * _singleAx_ or array of _singleAx_ objects <br />
    """
    subplotsargs = {'tight_layout': True, 'squeeze': False, 'nrows':rc//10, 'ncols':rc%10}
    subplotsargs.update(kwargs)
    fig, ax = plt.subplots(**subplotsargs)
    for i in range(ax.shape[0]):
        for j in range(ax.shape[1]):
            ax[i][j] = singleAx(ax[i][j])
    return(fig, np.array(ax))
