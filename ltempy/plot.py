import numpy as np
from .cielab import cielab_rgba
from . import constants as _

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

"""Contains wrappers for `matplotlib.pyplot`, tailored to the presentation of magnetic data.

The primary feature is the `singleAx` object, which extends the `maplotlib.axes.Axes` object,
adding methods for windowing data, quiver plots, CIELAB plots, and creating square insets and a colorwheel.

In addition, `cielab_cmap` and `rgba` provide utilities for general plotting.

The typical use case is to use `ltempy.subplots` to generate a figure and axes:

```python
# Generate data
X = numpy.linspace(-1, 1, 128)
x, y = numpy.meshgrid(X, X)
z = x + 1j*y
f = numpy.sin(z)

# Plot data
window = (.3, .7, .3, .7)
fig, [[ax1, ax2]] = ltempy.subplots(12)
ax1.set_axes(X, X)
ax1.inset(window)
ax1.imshow(numpy.abs(f)**2)
ax2.set_axes(ax1.x, ax1.y)
ax2.set_window(window)
ax2.rgba(f)
ax2.quiver(f, step=4)
plt.show()
```

**Note**:

This module is only loaded when `matplotlib` is available in the environment.
`matplotlib` is installed as a dependency when `ltempy` is installed with the `plot` extra:

```bash
pip install ltempy[plot]
```
"""

__all__ = ['singleAx', 'subplots', 'cielab_cmap', 'rgba']

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
	myax.rgba(data)
	plt.show()
	```

	More commonly, this class is returned by `ltempy.pyplotwrapper.subplots`.

	**Parameters**

	* **ax** : _matplotlib.axes.Axes_ <br />

	**Returns**

	* **singleAx** : _ltempy.plot.singleAx_ <br />
	"""
	def __init__(self, ax):
		self.ax = ax
		self.origin = 'lower'
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

		* **self** : _singleAx_
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

		* **self** : _singleAx_
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

		* **self** : _singleAx_
		"""
		self.ax.set_ylabel(ylabel, **kwargs)
		return(self)

	def set_xytitle(self, xlabel='', ylabel='', title='', **kwargs):
		"""Set the xlabel, ylabel, and title at the same time.

		Sets all three even if not all are given. Whatever you input will be applied to all three.

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

		* **self** : _singleAx_
		"""
		self.ax.set_xlabel(xlabel, **kwargs)
		self.ax.set_ylabel(ylabel, **kwargs)
		self.ax.set_title(title, **kwargs)
		return(self)

	def set_axes(self, x, y):
		"""Sets the x and y axes of the singleAx object, and can apply a window.

		Note that this can be used before or after `set_window()`, but make sure the two are in the same units.

		**Parameters**

		* **x** : _ndarray_ <br />
		The x-coordinates. Should be 1-dimensional.

		* **y** : _ndarray_ <br />
		The y-coordinates. Should be 1-dimensional.

		**Returns**

		* **self** : _singleAx_
		"""
		self.x = x
		self.y = y
		return(self)

	def set_window(self, window):
		"""Applies a window to the singleAx object.

		Note that this can be used before or after `set_axes()`, but make sure the two are in the same units.

		**Parameters**

		* **window** : _array-like_ <br />
		Format: `window = [xmin, xmax, ymin, ymax]`. Note that these are the x
		and y values, rather than their indices.

		**Returns**

		* **self** : _singleAx_
		"""
		self.xmin = window[0]
		self.xmax = window[1]
		self.ymin = window[2]
		self.ymax = window[3]
		return(self)

	def pre_plot(self, data, step=1):
		"""Utility function that applies the axes and window before plotting.

		If you want to use a plotting function from matplotlib, you can use this
		function to get the windowed axes and data:

		```python
		fig, axis = plt.subplots()
		ax = singleAx(axis)
		ax.set_axes(x, y)
		ax.set_window(window)
		x_windowed, y_windowed, data_windowed = ax.pre_plot(data)
		ax.ax.SomeOtherMatplotlibPlottingRoutine(x_windowed, y_windowed, data_windowed)
		plt.show()
		```

		**Parameters** :

		* **data** : _complex ndarray_ <br />
		The data to plot. Must be 2-dimensional.

		* **step** : _int, optional_ <br />
		data will be returned as `data[::step,::step]` - particularly useful for
		quiver plots. <br />
		Default is `step = 1`.

		**Returns**

		* **xout** : _ndarray_ <br />
		A 1darray with the windowed x coordinates.

		* **yout** : _ndarray_ <br />
		A 1darray with the windowed y coordinates.

		* **dout** : _ndarray_ <br />
		A 2darray with the windowed data.
		"""
		if self.x is None:
			self.x = np.linspace(0, 100, data.shape[1])
		if self.y is None:
			self.y = np.linspace(0, 100, data.shape[0])
		if self.xmin is None:
			self.xmin = self.x[0]
		if self.xmax is None:
			self.xmax = self.x[-1]
		if self.ymin is None:
			self.ymin = self.y[0]
		if self.ymax is None:
			self.ymax = self.y[-1]
		argxmin = np.argmin(np.abs(self.x - self.xmin))
		argxmax = np.argmin(np.abs(self.x - self.xmax))
		argymin = np.argmin(np.abs(self.x - self.ymin))
		argymax = np.argmin(np.abs(self.x - self.ymax))
		dout = data[argymin:argymax:step, argxmin:argxmax:step]
		xout = self.x[argxmin:argxmax:step]
		yout = self.y[argymin:argymax:step]
		return(xout, yout, dout)

	def imshow(self, data, step=1, **kwargs):
		"""Imshows the (windowed) data.

		**Parameters**

		* **data** : _ndarray_ <br />
		The data to be shown. Use the un-windowed data - the window will be
		applied automatically, if you have set one.

		* **step** : _int_ <br />
		data will be shown as `data[::step,::step]`. <br />
		Default is `step = 1`.

		* ****kwargs** <br />
		All other kwargs are passed on to `matplotlib.axes.Axes.imshow`.

		**Returns**

		* **self** : _singleAx_
		"""
		imshowargs = {'origin': self.origin}
		imshowargs.update(kwargs)
		x, y, d = self.pre_plot(data, step)
		if imshowargs['origin'] == 'lower':
			extent = [x[0], x[-1], y[0], y[-1]]
		elif imshowargs['origin'] == 'upper':
			extent = [x[0], x[-1], y[-1], y[0]]
		imshowargs.update({'extent': extent})
		imshowargs.update(kwargs)
		self.ax.imshow(d, **imshowargs)
		return(self)

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

		* **self** : _singleAx_
		"""
		if origin is None:
			origin = self.origin
		x, y, d = self.pre_plot(data, step)
		d = d.astype(complex)
		if origin == 'upper':
			d.imag *= -1
		self.ax.quiver(x, y, d.real, d.imag, **kwargs)
		return(self)

	def rgba(self, data, step=1, cmap=None, brightness='intensity', alpha='uniform', **kwargs):
		"""Show an rgba interpretation of complex data.

		**Parameters**

		* **data** : _complex ndarray_ <br />
		An array with the data to represent. Dtype may be complex or real - if real,
		the color will be uniform, and values will be represented by brightness.

		* **step** : _int_ <br />
		data will be shown as `data[::step,::step]`. <br />
		Default is `step = 1`.

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

		* **self** : _singleAx_
		"""
		imshowargs = {'origin': self.origin}
		imshowargs.update(kwargs)
		x, y, d = self.pre_plot(data, step)
		d = d.astype(complex)
		if imshowargs['origin'] == 'lower':
			extent = [x[0], x[-1], y[0], y[-1]]
		elif imshowargs['origin'] == 'upper':
			extent = [x[0], x[-1], y[-1], y[0]]
			d.imag *= -1
		imshowargs.update({'extent': extent})
		imshowargs.update(kwargs)
		self.ax.imshow(rgba(d, brightness=brightness, alpha=alpha, cmap=cmap), **imshowargs)
		return(self)

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

		* **self** : _singleAx_
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

	def colorwheel(self, res=128, scale=0.25, cmap=None, brightness='intensity', alpha='uniform', **kwargs):
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

		* **self** : _singleAx_
		"""
		imshowargs = {'origin': self.origin, 'zorder': 3}
		imshowargs.update(kwargs)
		X = np.linspace(-1, 1, res)
		x, y = np.meshgrid(X, X)
		z = x + 1j*y
		sel = np.abs(z) > 1
		z[sel] = 0
		colors = rgba(z, brightness=brightness, alpha=alpha, cmap=cmap)
		colors[:,:,0][sel] = 0
		colors[:,:,1][sel] = 0
		colors[:,:,2][sel] = 0
		xlims = self.ax.get_xlim()
		ylims = self.ax.get_ylim()
		self.ax.set_xlim(xlims)
		self.ax.set_ylim(ylims)
		self.ax.set_aspect('equal')
		if self.origin == 'lower':
			extent = [xlims[1] - scale * (xlims[1] - xlims[0]),
						xlims[1],
						ylims[0],
						ylims[0] + scale * (xlims[1] - xlims[0])]
		elif self.origin == 'upper':
			extent = [xlims[1] - scale * (xlims[1] - xlims[0]),
						xlims[1],
						ylims[0] + scale * (xlims[0] - xlims[1]),
						ylims[0]]
		imshowargs.update(kwargs)
		self.ax.imshow(colors, extent=extent, **imshowargs)
		return(self)

def subplots(rc=11, **kwargs):
		"""Creates a (fig, [[ax]]) instance but replaces ax with singleAx.

		Behaves almost identically to matplotlib.pyplot.subplots(), but replaces each
		`matplotlib.axes.Axes` object with a `wsp_tools.pyplotwrapper.singleAx`
		object.

		Each `wsp_tools.pyplotwrapper.singleAx` object in turn behaves just like a
		normal `Axes` object, but with added methods.

		Note: two default kwargs are set. `'tight_layout': True` and `'squeeze': False`.

		**Parameters**

		* **rc** : _int_ <br />
		First digit - nrows. Second digit - ncols. <br />
		Default is `rc = 11`.

		* ****kwargs** <br />
		All other kwargs are passed on to `matplotlib.axes.Axes.subplots`.

		**Returns**

		* **fig** : _Figure_ <br />

		* **ax** : _singleAx_ or array of _singleAx_ objects <br />
		"""
		subplotsargs = {'tight_layout': True, 'squeeze': False}
		subplotsargs.update(kwargs)
		fig, ax = plt.subplots(nrows=rc//10, ncols=rc%10, **subplotsargs)
		for i in range(ax.shape[0]):
			for j in range(ax.shape[1]):
				ax[i][j] = singleAx(ax[i][j])
		return(fig, np.array(ax))

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
	angles = np.linspace(0,2*_.pi,samples)
	cvals = np.exp(1j*angles).reshape(1, samples)
	rgbavals = cielab_rgba(cvals).squeeze()/255
	cmap = ListedColormap(rgbavals)
	return(cmap)

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
		out = cielab_rgba(mode, brightness, alpha)
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
