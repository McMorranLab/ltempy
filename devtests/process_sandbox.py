# %%
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import ltempy as lp
from warnings import warn

# %%


def _extend_and_fill_mirror(data):
    ds0 = data.shape[0]
    ds1 = data.shape[1]
    bdata = np.zeros((3 * ds0, 3 * ds1), dtype=data.dtype)
    bdata[0:ds0, 0:ds1] = data[::-1, ::-1]
    bdata[0:ds0, ds1:2*ds1] = data[::-1, :]
    bdata[0:ds0, 2*ds1:3*ds1] = data[::-1, ::-1]
    bdata[ds0:2*ds0, 0:ds1] = data[:, ::-1]
    bdata[ds0:2*ds0, ds1:2*ds1] = data
    bdata[ds0:2*ds0, 2*ds1:3*ds1] = data[:, ::-1]
    bdata[2*ds0:3*ds0, 0:ds1] = data[::-1, ::-1]
    bdata[2*ds0:3*ds0, ds1:2*ds1] = data[::-1]
    bdata[2*ds0:3*ds0, 2*ds1:3*ds1] = data[::-1, ::-1]
    return(bdata)


def high_pass(data, cutoff=7, gaussian=False, dx=1, dy=1):
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


def low_pass(data, cutoff=10, gaussian=False, dx=1, dy=1):
    X = np.fft.fftfreq(data.shape[1], dx)
    Y = np.fft.fftfreq(data.shape[0], dy)
    x, y = np.meshgrid(X, Y)

    g = np.exp(-(x**2+y**2)/2/cutoff**2)
    if not gaussian:
        g[x**2 + y**2 < cutoff**2] = 1
        g[x**2 + y**2 >= cutoff**2] = 0
    else:
        warn(
            "Gaussian high-pass is being deprecated and will be removed in a future version of ltempy",
            DeprecationWarning,
            stacklevel=2
        )

    Fdata = np.fft.fft2(data)
    FFdata = np.fft.ifft2(g * Fdata)
    return(FFdata)


def gaussian_blur(data, blur_radius=10):
    ds0, ds1 = data.shape[0], data.shape[1]
    bdata = _extend_and_fill_mirror(data)
    X = np.fft.fftfreq(bdata.shape[1], 1)
    Y = np.fft.fftfreq(bdata.shape[0], 1)
    x, y = np.meshgrid(X, Y)

    g = np.exp(-2 * np.pi**2 * blur_radius**2 * (x**2 + y**2))

    Fdata = np.fft.fft2(bdata)
    FFdata = np.fft.ifft2(g * Fdata)
    return(FFdata[ds0:2*ds0, ds1:2*ds1])


# %%
X = np.linspace(0, 127, 128)
dx = X[1] - X[0]
x, y = np.meshgrid(X, X)
f1 = 1  # cycles per unit; f1 = 1 / 128 cycles per px
f2 = 2  # = 2 / 128 cycles per px
f1px = f1 * 1 / 128
f2px = f2 * 1 / 128
# data = np.cos(2 * np.pi * f1 * x)
# data += np.cos(2 * np.pi * f2 * y)
data = np.zeros_like(x)
selx = (x/8) % 2 > 1
sely = (y/8) % 2 > 1
data[selx ^ sely] = 1
data += np.sin(2 * np.pi * 3 / 128 * x)
data += np.random.random((128, 128))

blur_radius = 1  # cycles per pixel -> pixels per cycle

high = high_pass(data, cutoff=5 / 128)
low = low_pass(data, cutoff=0.4)
gauss = gaussian_blur(data, blur_radius=blur_radius)

fig, [[ax1, ax2], [ax3, ax4]] = lp.subplots(22, dpi=150)
ax1.set_title("input")
ax2.set_title("gaussian blur")
ax3.set_title("high pass")
ax4.set_title("low pass")
ax1.imshow(data)
ax2.imshow(np.real(gauss))
ax3.imshow(np.real(high))
ax4.imshow(np.real(low))
plt.show()


# %%
class ndap(np.ndarray):
    def __new__(cls, data, x=None, y=None):
        obj = np.asarray(data, dtype=data.dtype).copy().view(cls)
        obj.x = x
        obj.y = y
        return(obj)

    def __array_finalize__(self, obj):
      if obj is None: return
      self.x = getattr(self, 'x', None)
      self.y = getattr(self, 'y', None)
      self.isComplex = np.iscomplexobj(obj)
      if not x is None:
          try:
              sel = obj.shape[1] != x.shape[0] or len(x.shape) != 1
          except:
              raise Exception("x must be a 1-dimensional numpy.ndarray. ")
          if sel:
              raise Exception(
                  "x shape must match the data's 1st axis. ")
          self.x = x
          self.dx = x[1] - x[0]
      else:
          self.x = np.arange(obj.shape[1])
          self.dx = 1
      if not y is None:
          try:
              sel = obj.shape[0] != y.shape[0] or len(y.shape) != 1
          except:
              raise Exception("y must be a 1-dimensional numpy.ndarray. ")
          if sel:
              raise Exception(
                  "y shape must match the data's 1st axis. ")
          self.y = y
          self.dy = y[1] - y[0]
      else:
          self.y = np.arange(obj.shape[0])
          self.dy = 1
    
    def asdf(self):
      self[:, :] = 40
      return(self)


# %%
test = ndap(x, X, X)

test = test**2
print(test)

test.asdf()

print(test)


# %%
class RealisticInfoArray(np.ndarray):

    def __new__(cls, input_array, info=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        if info is None:
          info = "asdf"
        obj.info = info
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        self.info = getattr(obj, 'info', None)

# %%
test = RealisticInfoArray(x)
print(test.info)
test = test**2
print(test.info)