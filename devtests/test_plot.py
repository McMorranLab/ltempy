try:
	import matplotlib.pyplot as plt
	import ltempy as lp
	import numpy as np
except:
	raise ImportError('To test ltempy.plot, you need to install the matplotlib package. ')

X = np.linspace(-1, 1, 1024)
x, y = np.meshgrid(X, X)
z = x + 1j*y

plt.title("rgba_cmap(cmap=None)")
plt.imshow(lp.rgba(z))
plt.show()

opts = ['uniform', 'intensity', 'amplitude']

for brightness in opts:
	for alpha in opts:
		plt.title("cmap=viridis, brightness: {}, alpha: {}".format(brightness, alpha))
		plt.imshow(lp.rgba(z, cmap='viridis', brightness = brightness, alpha = alpha))
		plt.show()

ccmap = lp.cielab_cmap()
plt.title("using cielab as a mpl cmap. ")
plt.imshow(np.angle(z), cmap=ccmap)
plt.show()

window = (.3, .4, .3, .4)

fig, [[ax1, ax2]] = lp.subplots(12)
fig.suptitle("origin: upper")
ax1.origin = 'upper'
ax2.origin = ax1.origin
ax1.set_xytitle("x", "y", "xytitle")
ax1.set_axes(X, X)
ax1.inset(window)
ax1.imshow(np.abs(z))
ax2.set_xlabel("x")
ax2.set_title("title")
ax2.set_ylabel("y")
ax2.set_axes(ax1.x, ax1.y)
ax2.set_window(window)
ax2.rgba(z)
ax2.quiver(z, step = 12)
ax2.colorwheel()
plt.show()


fig, [[ax1, ax2]] = lp.subplots(12)
fig.suptitle("origin: lower")
ax1.origin = 'lower'
ax2.origin = ax1.origin
ax1.set_xytitle("x", "y", "xytitle")
ax1.set_axes(X, X)
ax1.inset(window)
ax1.imshow(np.abs(z))
ax2.set_xlabel("x")
ax2.set_title("title")
ax2.set_ylabel("y")
ax2.set_axes(ax1.x, ax1.y)
ax2.set_window(window)
ax2.rgba(z)
ax2.quiver(z, step = 12)
ax2.colorwheel()
plt.show()
