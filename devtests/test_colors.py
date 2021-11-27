import ltempy as lp
import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(-1, 1, 100)
Y = np.linspace(-0.5, 0.5, 50)
x, y = np.meshgrid(X, Y)
z = x + 1j * y

# rgba(mode, cmap = None, brightness = 'intensity', alpha = 'uniform')
print("testing rgba()")
opts = ['uniform', 'amplitude', 'intensity']
for brightness in opts:
	for alpha in opts:
		data = lp.rgba(z, brightness = brightness, alpha = alpha)
		plt.title("brightness: {}, alpha: {}".format(brightness, alpha))
		plt.imshow(data)
		plt.show()
		data = lp.rgba(z, cmap='viridis', brightness = brightness, alpha = alpha)
		plt.title("brightness: {}, alpha: {}, cmap='viridis'".format(brightness, alpha))
		plt.imshow(data)
		plt.show()

# cielab_cmap(samples=256)
print("testing cielab_cmap()")
ccmap = lp.cielab_cmap(128)
plt.imshow(np.angle(z), cmap=ccmap)
plt.show()

# # cielab_rgba(data, brightness = 'intensity', alpha = 'uniform')
print("testing cielab_rgba()")
for brightness in opts:
	for alpha in opts:
		plt.title("brightness: {}, alpha: {}".format(brightness, alpha))
		plt.imshow(lp.rgba(z, brightness = brightness, alpha = alpha))
		plt.show()
