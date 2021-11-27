import ltempy as lp
from ltempy import constants as _
import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(-2e-6, 2e-6, 512)
Y = np.linspace(-2e-6, 2e-6, 256)
x, y = np.meshgrid(X, X)
dx = X[1] - X[0]
dy = Y[1] - Y[0]
Lx = X[-1] - X[0]

sig = .8e-6
M_s = 10 * 10**3
f = 10
thickness = 60e-9

df = 1e-3
C_s = 2.7e-3
divangle = 1e-5

mx = np.zeros_like(x)
my = M_s * np.cos(f * 2 * np.pi * x / Lx) * np.exp(-(x**2 + y**2) / 2 / sig**2)
mz = np.sqrt(np.max(mx**2 + my**2) - mx**2 - my**2)

B0 = lp.B_from_mag(mx, my, mz, dx = dx, dy = dy)
B120 = lp.B_from_mag(mx, my, mz, dx = dx, dy = dy, z = 120e-9)
A0 = lp.A_from_mag(mx, my, mz, dx = dx, dy = dy)
A120 = lp.A_from_mag(mx, my, mz, dx = dx, dy = dy, z = 120e-9)

ab_phase = lp.ab_phase(mx, my, mz, dx = dx, dy = dy, thickness = thickness)
ideal_phase = - 2 * _.e * thickness * M_s / _.hbar / _.c / f * Lx * np.sin(f * 2 * _.pi * x / Lx)
ind = lp.ind_from_mag(mx, my, mz, dx = dx, dy = dy)
img_tie_from_mag = lp.img_from_mag(mx, my, mz, dx = dx, dy = dy, defocus = 1e-3)
img_tie_from_phase = lp.img_from_phase(ab_phase, dx = dx, dy = dy, defocus = 1e-3)
img_tie_from_ideal_phase = lp.img_from_phase(ideal_phase, dx = dx, dy = dy, defocus = 1e-3)
img_propagate = lp.propagate(np.exp(1j*ab_phase), dx = dx, dy = dy, defocus = 1e-3)
img_propagate_ideal = lp.propagate(np.exp(1j*ideal_phase), dx = dx, dy = dy, defocus = 1e-3)

fig, [[ax]] = lp.subplots()
ax.set_title("2d magnetization")
ax.set_axes(X, X)
ax.colorbar(ax.imshow(mz))
ax.quiver(mx + 1j*my, step=8, pivot='mid')
plt.show()

fig, [[ax]] = lp.subplots()
ax.set_title("B at z=0")
im = ax.imshow(B0[2])
ax.quiver(B0[0] + 1j*B0[1], step=8, pivot='mid')
ax.colorbar(im)
plt.show()

fig, [[ax]] = lp.subplots()
ax.set_title("B at z=120e-9")
im = ax.imshow(B120[2])
ax.quiver(B120[0] + 1j*B120[1], step=8, pivot='mid')
ax.colorbar(im)
plt.show()

fig, [[ax]] = lp.subplots()
ax.set_title("A at z=0")
im = ax.imshow(A0[2])
ax.quiver(A0[0] + 1j*A0[1], step=8, pivot='mid')
ax.colorbar(im)
plt.show()

fig, [[ax]] = lp.subplots()
ax.set_title("A at z=120e-9")
im = ax.imshow(A120[2])
ax.quiver(A120[0] + 1j*A120[1], step=8, pivot='mid')
ax.colorbar(im)
plt.show()

fig, [[ax]] = lp.subplots()
ax.set_title("2d ab_phase")
ax.set_axes(X, X)
im = ax.imshow(ab_phase)
ax.colorbar(im)
plt.show()

fig, [[ax]] = lp.subplots()
ax.set_title("2d ideal_phase")
ax.set_axes(X, X)
im = ax.imshow(ideal_phase)
ax.colorbar(im)
plt.show()

fig, [[ax]] = lp.subplots()
ax.set_title("phases compared (ideal has no envelope)")
ax.ax.plot(ab_phase[256,])
ax.ax.plot(ideal_phase[256,])
ax.ax.legend(["ab_phase", "ideal_phase"])
plt.show()

fig, [[ax]] = lp.subplots()
ax.set_title("ind")
im = ax.imshow(lp.rgba(ind[0] + 1j*ind[1]))
ax.quiver(ind[0] + 1j*ind[1], step=8, pivot='mid')
ax.colorbar(im)
plt.show()

fig, [[ax]] = lp.subplots()
ax.set_title("img_tie_from_mag")
im = ax.imshow(img_tie_from_mag)
ax.colorbar(im)
plt.show()

fig, [[ax]] = lp.subplots()
ax.set_title("img_tie_from_phase")
im = ax.imshow(img_tie_from_phase)
ax.colorbar(im)
plt.show()

fig, [[ax]] = lp.subplots()
ax.set_title("img_tie_from_ideal_phase")
im = ax.imshow(img_tie_from_ideal_phase)
ax.colorbar(im)
plt.show()

fig, [[ax]] = lp.subplots()
ax.set_title("imgs compared (ideal has no envelope)")
ax.ax.plot(img_tie_from_phase[256,])
ax.ax.plot(img_tie_from_ideal_phase[256,])
ax.ax.legend(["from ab", "from ideal"])
plt.show()

fig, [[ax]] = lp.subplots()
ax.set_title("img_propagate")
im = ax.imshow(np.abs(img_propagate)**2)
ax.colorbar(im)
plt.show()

fig, [[ax]] = lp.subplots()
ax.set_title("img_propagate_ideal")
im = ax.imshow(np.abs(img_propagate_ideal)**2)
ax.colorbar(im)
plt.show()

fig, [[ax]] = lp.subplots()
ax.set_title("propagated compared (ideal has no env)")
ax.ax.plot(np.abs(img_propagate[256])**2)
ax.ax.plot(np.abs(img_propagate_ideal[256])**2)
plt.show()

################
X = np.linspace(-2e-6, 2e-6, 512)
Y = np.linspace(-2e-6, 2e-6, 256)
Z = np.linspace(-30e-6, 30e-6, 3)
x, y, z = np.meshgrid(X, X, Z)
dx = X[1] - X[0]
dy = Y[1] - Y[0]
Lx = X[-1] - X[0]

sig = .8e-6
M_s = 10 * 10**3
f = 10
thickness = 60e-9

df = 1e-3
C_s = 2.7e-3
divangle = 1e-5

mx = np.zeros_like(x)
my = M_s * np.cos(f * 2 * np.pi * x / Lx) * np.exp(-(x**2 + y**2) / 2 / sig**2)
mz = np.sqrt(np.max(mx**2 + my**2) - mx**2 - my**2)

ab_phase = lp.ab_phase(mx, my, mz, dx = dx, dy = dy, thickness = thickness)
ideal_phase = - 2 * _.e * thickness * M_s / _.hbar / _.c / f * Lx * np.sin(f * 2 * _.pi * x / Lx)
ind = lp.ind_from_mag(mx, my, mz, dx = dx, dy = dy)
img_tie_from_mag = lp.img_from_mag(mx, my, mz, dx = dx, dy = dy, defocus = 1e-3)

fig, [[ax]] = lp.subplots()
ax.set_title("3d magnetization")
ax.set_axes(X, X)
im = ax.imshow(mz[:,:,0])
ax.colorbar(im)
ax.quiver(mx[:,:,0] + 1j*my[:,:,0], step=8, pivot='mid')
plt.show()

fig, [[ax]] = lp.subplots()
ax.set_title("3d ab_phase")
ax.set_axes(X, X)
im = ax.imshow(ab_phase[:,:,0])
ax.colorbar(im)
plt.show()

fig, [[ax]] = lp.subplots()
ax.set_title("3d ideal_phase")
ax.set_axes(X, X)
im = ax.imshow(ideal_phase[:,:,0])
ax.colorbar(im)
plt.show()

fig, [[ax]] = lp.subplots()
ax.set_title("3d phases compared (ideal has no envelope)")
ax.ax.plot(ab_phase[256,...,0])
ax.ax.plot(ideal_phase[256,...,0])
ax.ax.legend(["ab_phase", "ideal_phase"])
plt.show()

fig, [[ax]] = lp.subplots()
ax.set_title("3d ind")
im = ax.imshow(lp.rgba(ind[0,...,0] + 1j*ind[1,...,0]))
ax.colorbar(im)
ax.quiver(ind[0,...,0] + 1j*ind[1,...,0], step=8, pivot='mid')
plt.show()

fig, [[ax]] = lp.subplots()
ax.set_title("3d img_tie_from_mag")
im = ax.imshow(img_tie_from_mag)
ax.colorbar(im)
plt.show()
