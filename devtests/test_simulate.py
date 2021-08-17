import ltempy as lp
import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(-50e-9, 50e-9, 128)
x2, y2 = np.meshgrid(X, X)
dx = X[1] - X[0]

Z = np.linspace(-60e-9, 60e-9, 16)
x3, y3, z3 = np.meshgrid(X, X, Z)

m2 = lp.jchessmodel(x2, y2)
m3 = lp.jchessmodel(x3, y3, z3)
phase2 = lp.ab_phase(m2[0], m2[1], m2[2], dx, dx)
phase3 = lp.ab_phase(m3[0], m3[1], m3[2], dx, dx)
B = lp.B_from_mag(m2[0], m2[1], m2[2], dx, dx)
A = lp.A_from_mag(m2[0], m2[1], m2[2], dx, dx)
img2 = lp.img_from_mag(m2[0], m2[1], m2[2], dx, dx, 1e-3)
img3 = lp.img_from_mag(m3[0], m3[1], m3[2], dx, dx, 1e-3)
img_from_phase = lp.img_from_phase(phase2, dx, dx, 1e-3)
ind_from_mag2 = lp.ind_from_mag(m2[0], m2[1], m2[2], dx, dx)
ind_from_mag3 = lp.ind_from_mag(m3[0], m3[1], m3[2], dx, dx)
propd = lp.propagate(np.exp(1j*phase2), dx, dx, defocus = .1e-6)

fig, [[ax]] = lp.subplots()
ax.set_title("2d mag")
ax.rgba(m2[0] + 1j*m2[1])
ax.quiver(m2[0] + 1j*m2[1], step=4)
ax.colorwheel()
plt.show()

fig, [[ax]] = lp.subplots()
ax.set_title("3d mag (z = 0)")
ax.rgba(m3[0][:,:,8] + 1j*m3[1][:,:,8])
ax.colorwheel()
plt.show()

fig, [[ax]] = lp.subplots()
ax.set_title("phase of m2, max: {}".format(np.max(np.abs(phase2))))
ax.imshow(phase2)
plt.show()

fig, [[ax]] = lp.subplots()
ax.set_title("phase of m3 (z=0), max: {}".format(np.max(np.abs(phase2))))
ax.imshow(phase3[:,:,8])
plt.show()

fig, [[ax]] = lp.subplots()
ax.set_title("B of m2 (z = 0)")
ax.imshow(B[2])
ax.quiver(B[0] + 1j*B[1], step = 4)
plt.show()

fig, [[ax]] = lp.subplots()
ax.set_title("A of m2 (z = 0)")
ax.imshow(A[2])
ax.quiver(A[0] + 1j*A[1], step = 4)
plt.show()

fig, [[ax]] = lp.subplots()
ax.set_title("ind_from_mag2, max: {}".format(np.max(np.sqrt(ind_from_mag2[0]**2 + ind_from_mag2[1]**2))))
ax.rgba(ind_from_mag2[0] + ind_from_mag2[1] * 1j)
ax.colorwheel()
plt.show()

fig, [[ax]] = lp.subplots()
ax.set_title("ind_from_mag3")
ax.rgba(ind_from_mag3[0] + ind_from_mag3[1] * 1j)
ax.colorwheel()
plt.show()

fig, [[ax]] = lp.subplots()
ax.set_title("img from m2")
ax.imshow(np.abs(img2)**2)
plt.show()

fig, [[ax]] = lp.subplots()
ax.set_title("img from m3")
ax.imshow(np.abs(img3)**2)
plt.show()

fig, [[ax]] = lp.subplots()
ax.set_title("img from phase2")
ax.imshow(np.abs(img_from_phase)**2)
plt.show()

fig, [[ax]] = lp.subplots()
ax.set_title("propagated")
ax.imshow(np.abs(propd)**2)
plt.show()
