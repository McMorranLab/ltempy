import ltempy as lp
from ltempy import constants as _
import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(-200e-9, 200e-9, 256)
x2, y2 = np.meshgrid(X, X)
dx = X[1] - X[0]

Z = np.linspace(-60e-9, 60e-9, 16)
x3, y3, z3 = np.meshgrid(X, X, Z)

df = .2e-3
sig = 2e-9

M_s = 1000 * 10**3


# m2 = lp.jchessmodel(x2, y2, aa=3, ak=2e7)
# m3 = m2[...,np.newaxis]
mx = np.zeros_like(x2)
my = M_s * np.cos(2 * np.pi * x2 / (np.max(x2) - np.min(x2)))
mz = np.sqrt(np.max(mx**2 + my**2) - mx**2 - my**2)
m2 = np.array([mx, my, mz])
m3 = np.array([mx, my, mz])[...,np.newaxis]

phase2 = lp.ab_phase(m2[0], m2[1], m2[2], dx = dx, dy = dx)
ideal_phase = - 2 * _.e * 60e-9 * M_s * (np.max(x2) - np.min(x2)) / _.hbar / _.c * np.sin(2 * _.pi * x2 / (np.max(x2) - np.min(x2)))
phase3 = lp.ab_phase(m3[0], m3[1], m3[2], dx = dx, dy = dx)
B = lp.B_from_mag(m2[0], m2[1], m2[2], dx, dx)
A = lp.A_from_mag(m2[0], m2[1], m2[2], dx, dx)
img2 = lp.img_from_mag(m2[0], m2[1], m2[2], dx = dx, dy = dx, defocus = df)
img3 = lp.img_from_mag(m3[0], m3[1], m3[2], dx = dx, dy = dx, defocus = df)
img_from_phase = lp.img_from_phase(phase2, dx = dx, dy = dx, defocus = df)
ind_from_mag2 = lp.ind_from_mag(m2[0], m2[1], m2[2], dx = dx, dy = dx)
ind_from_mag3 = lp.ind_from_mag(m3[0], m3[1], m3[2], dx = dx, dy = dx)
mode = np.exp(1j * phase2)
propd = lp.propagate(mode = mode, dx = dx, dy = dx, defocus = df)

fig, [[ax]] = lp.subplots()
ax.set_title("2d mag")
# ax.imshow(m2[2])
ax.rgba(m2[0] + 1j*m2[1])
# ax.quiver(m2[0] + 1j*m2[1], step=4, color='white')
ax.colorwheel()
plt.show()

plt.title("phase")
plt.imshow(phase2)
plt.show()

plt.plot(propd[128])
plt.plot(img2[128])
plt.legend(['propd', 'from mag'])
plt.show()
#
plt.title("propagated")
plt.imshow(propd)
plt.show()
#
# plt.title("img from mag")
# plt.imshow(img2)
# plt.show()
# #
# plt.title("img from phase")
# plt.imshow(img_from_phase)
# plt.show()

# fig, [[ax]] = lp.subplots()
# ax.set_title("2d mag")
# # ax.imshow(m2[2])
# ax.rgba(m2[0] + 1j*m2[1])
# ax.quiver(m2[0] + 1j*m2[1], step=4)
# ax.colorwheel()
# plt.show()
#
# # fig, [[ax]] = lp.subplots()
# # ax.set_title("3d mag (z = 0)")
# # ax.rgba(m3[0][:,:,8] + 1j*m3[1][:,:,8])
# # ax.colorwheel()
# # plt.show()
#
# # fig, [[ax]] = lp.subplots()
# # ax.set_title("phase of m2, max: {}".format(np.max(np.abs(phase2))))
# # ax.imshow(phase2)
# # plt.show()
# #
# # fig, [[ax]] = lp.subplots()
# # ax.set_title("phase of m3 (z=0), max: {}".format(np.max(np.abs(phase2))))
# # ax.imshow(phase3[:,:,8])
# # plt.show()
# #
fig, [[ax1, ax2, ax3]] = lp.subplots(13)
ax1.set_title("mag 2")
ax1.imshow(m2[2])
ax1.quiver(m2[0] + 1j*m2[1], step=4)
ax2.set_title("B of m2 (z = 0)")
ax2.imshow(B[2])
ax2.quiver(B[0] + 1j*B[1], step = 4)
ax3.set_title("A of m2 (z = 0)")
ax3.imshow(A[2])
ax3.quiver(A[0] + 1j*A[1], step = 4)
plt.show()
# #
fig, [[ax]] = lp.subplots()
ax.set_title("A of m2 (z = 0)")
ax.imshow(A[2])
ax.quiver(A[0] + 1j*A[1], step = 4)
plt.show()
# #
# # fig, [[ax]] = lp.subplots()
# # ax.set_title("ind_from_mag2, max: {}".format(np.max(np.sqrt(ind_from_mag2[0]**2 + ind_from_mag2[1]**2))))
# # ax.rgba(ind_from_mag2[0] + ind_from_mag2[1] * 1j)
# # ax.colorwheel()
# # plt.show()
# #
# # fig, [[ax]] = lp.subplots()
# # ax.set_title("ind_from_mag3")
# # ax.rgba(ind_from_mag3[0] + ind_from_mag3[1] * 1j)
# # ax.colorwheel()
# # plt.show()
#
# # fig, [[ax]] = lp.subplots()
# # ax.set_title("img from m2")
# # ax.imshow(img2)
# # plt.show()
#
# # fig, [[ax]] = lp.subplots()
# # ax.set_title("img from m3")
# # ax.imshow(np.abs(img3)**2)
# # plt.show()
# #
# # fig, [[ax]] = lp.subplots()
# # ax.set_title("img from phase2")
# # ax.imshow(img_from_phase)
# # plt.show()
# plt.imshow(img2)
# plt.colorbar()
# plt.show()
# #
# # fig, [[ax]] = lp.subplots()
# # ax.set_title("propagated")
# # ax.imshow(np.abs(propd)**2)
# # plt.show()
#
# plt.imshow(propd)
# plt.colorbar()
# plt.show()
