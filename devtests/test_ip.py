import ltempy as wt
import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(-50e-9, 50e-9, 128)
x, y = np.meshgrid(X, X)
m = wt.jchessmodel(x, y, n=1)
mx = wt.ndap(m[0])
my = wt.ndap(m[1])
mz = wt.ndap(m[2])
plt.imshow(wt.rgba(mx + 1j*my))
plt.show()

sigma = 5
mx.low_pass(sigma=sigma)
my.low_pass(sigma=sigma)
plt.imshow(wt.rgba(mx + 1j*my))
plt.show()

X = np.linspace(-50e-9, 50e-9, 128)
x, y = np.meshgrid(X, X)
m = wt.jchessmodel(x, y, n=1)
mx = wt.ndap(m[0])
my = wt.ndap(m[1])
mz = wt.ndap(m[2])
plt.imshow(wt.rgba(mx + 1j*my))
plt.show()

sigma = 5
mx.low_pass(sigma=sigma, tophat=True)
my.low_pass(sigma=sigma, tophat=True)
plt.imshow(wt.rgba(mx + 1j*my))
plt.show()
