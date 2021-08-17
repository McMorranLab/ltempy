import ltempy as lp
import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(-50e-9, 50e-9, 128)
x2, y2 = np.meshgrid(X, X)
dx = X[1] - X[0]

m2 = lp.jchessmodel(x2, y2)
img2 = lp.img_from_mag(m2[0], m2[1], m2[2], dx, dx, 1e-3)
phase_from_mag = lp.ab_phase(m2[0], m2[1], m2[2], dx, dx)
phase = lp.phase_from_img(img2, defocus=1e-3, dx = dx, dy = dx)
ind = lp.ind_from_img(img2, defocus=1e-3, dx = dx, dy = dx)
ind_from_phase = lp.ind_from_phase(phase)

dummy = {'data':img2, 'pixelSize' : [dx, dx], 'pixelUnit':['m','m']}
img = lp.sitie_image(dummy)
img.set_units(1e3, 'mm')
img.data.clip_data().shift_pos()
img.reconstruct()
print(img.validate(dx = dx, defocus = 1000e-6))


fig, [[ax1, ax2]] = lp.subplots(12)
ax1.set_axes(img.x, img.y)
ax1.set_xytitle("x ({})".format(img.x_unit), "y ({})".format(img.y_unit), "img data")
ax1.imshow(img.data)
ax2.set_title("reconstructed induction")
ax2.rgba(img.Bx + 1j*img.By)
plt.show()

# plt.title("phase from mag")
# plt.imshow(phase_from_mag)
# plt.colorbar()
# plt.show()
#
# plt.title("phase from sitie")
# plt.imshow(phase)
# plt.colorbar()
# plt.show()
#
# fig, [[ax]] = lp.subplots()
# ax.set_title("ind from sitie, max: {}".format(np.max(np.sqrt(ind[0]**2 + ind[1]**2))))
# ax.rgba(ind[0]+1j*ind[1])
# ax.quiver(ind[0] + 1j*ind[1], step=4)
# plt.show()
#
# fig, [[ax]] = lp.subplots()
# ax.set_title("ind from phase, max: {}".format(np.max(np.sqrt(ind[0]**2 + ind[1]**2))))
# ax.rgba(ind_from_phase[0]+1j*ind_from_phase[1])
# ax.quiver(ind_from_phase[0] + 1j*ind_from_phase[1], step=4)
# plt.show()
