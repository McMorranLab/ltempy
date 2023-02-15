import matplotlib.pyplot as plt
import ltempy as lp
import numpy as np

X = np.random.random((4, 4))
fig, [[ax]] = lp.subplots()
ax.imshow(X)
plt.show()

# Testing example from docs
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
ax1.colorwheel()
ax2.shift = np.pi / 2
ax2.set_axes(ax1.x, ax1.y)
ax2.set_window(window)
ax2.cielab(f)
ax2.colorwheel()
ax2.quiver(f, step=4)
plt.show()

#######

X = np.linspace(0, 1, 256)
Y = np.linspace(0, 1.5, 256 + 128)
x, y = np.meshgrid(X, Y)
z = x + 1j*y

window = (0.5, 0.75, 0.5, 0.75)
fig, [[ax1, ax2, ax3]] = lp.subplots(13)
ax1.origin = 'upper'
ax1.set_title("title")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_axes(X, Y)
ax1.cielab(z)
ax1.inset(window)
ax1.colorwheel()

ax2.set_xytitle("x", "y", "title")
ax2.set_axes(X, Y)
ax2.imshow(lp.rgba(z))
ax2.quiver(z, step=12)
ax2.colorwheel()

ax3.set_axes(X, Y)
ax3.set_window(window)
im3 = ax3.imshow(np.abs(z), cmap='hot')
ax3.quiver(z, step=8)
ax3.colorbar(im3)
plt.show()
