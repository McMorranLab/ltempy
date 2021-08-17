import ltempy as lp
import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(-1, 1, 100)
x, y = np.meshgrid(X, X)
z = x + 1j * y

opts = ['uniform', 'intensity', 'amplitude']

for brightness in opts:
	for alpha in opts:
		plt.title("brightness: {}, alpha: {}".format(brightness, alpha))
		plt.imshow(lp.cielab_rgba(z, brightness = brightness, alpha = alpha))
		plt.show()
