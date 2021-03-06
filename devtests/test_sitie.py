import ltempy as lp
import numpy as np
import matplotlib.pyplot as plt
import ncempy.io.ser as ser

fname = "img009_worms_gone_1.ser"
file = ser.serReader(fname)
img = lp.SITIEImage(file)
img.data.clip_data().high_pass(cutoff=10 / 2048)
img.reconstruct(df=1e-3)

window = [2.5, 3.0, 1.75, 2.25]
fig, [[ax1, ax2]] = lp.subplots(12, dpi=150)
ax1.set_axes(1e6 * img.x, 1e6 * img.y)
ax2.set_axes(1e6 * img.x, 1e6 * img.y)
ax1.inset(window)
ax1.imshow(img.data)
ax2.set_window(window)
ax2.cielab(img.Bx + 1j*img.By, brightness='amplitude')
ax2.quiver(img.Bx + 1j*img.By, step=8, color='white', pivot='mid')
plt.show()

X = np.linspace(-50e-9, 50e-9, 128)
Y = np.linspace(-50e-9, 50e-9, 128)
x2, y2 = np.meshgrid(X, Y)
dx = X[1] - X[0]

m2 = lp.jchessmodel(x2, y2)
img2 = lp.img_from_mag(m2[0], m2[1], m2[2], dx, dx, 1e-3)
phase_from_mag = lp.ab_phase(m2[0], m2[1], m2[2], dx, dx)
phase = lp.sitie(img2, defocus=1e-3, dx=dx, dy=dx)
ind = lp.ind_from_phase(phase, thickness=60e-9)
ind_from_phase = lp.ind_from_phase(phase)

dummy = {'data': img2[::-1, :], 'pixelSize': [dx, dx], 'pixelUnit': ['m', 'm']}
img = lp.SITIEImage(dummy)
img.set_units(1e3, 'mm')
img.data.clip_data().shift_pos()
img.reconstruct()
print("1 / img.validate: ", 1 / img.validate(defocus=500e-6, divangle=1e-5))

fig, [[ax]] = lp.subplots()
ax.set_title("magnetization")
ax.cielab(m2[0] + 1j*m2[1])
ax.colorwheel()
plt.show()

fig, [[ax1, ax2]] = lp.subplots(12)
ax1.set_axes(img.x, img.y)
ax1.set_xytitle("x ({})".format(img.x_unit),
                "y ({})".format(img.y_unit), "img (simulated)")
ax1.imshow(img.data)
ax2.set_title("reconstructed induction")
ax2.cielab(img.Bx + 1j*img.By)
ax2.colorwheel()
plt.show()

plt.title("phase from mag")
plt.imshow(phase_from_mag)
plt.colorbar()
plt.show()

plt.title("phase from sitie")
plt.imshow(phase)
plt.colorbar()
plt.show()

fig, [[ax]] = lp.subplots()
ax.set_title("ind from sitie, max: {}".format(
    np.max(np.sqrt(ind[0]**2 + ind[1]**2))))
ax.cielab(ind[0]+1j*ind[1])
ax.quiver(ind[0] + 1j*ind[1], step=4)
plt.show()

fig, [[ax]] = lp.subplots()
ax.set_title("ind from phase, max: {}".format(
    np.max(np.sqrt(ind[0]**2 + ind[1]**2))))
ax.cielab(ind_from_phase[0]+1j*ind_from_phase[1])
ax.quiver(ind_from_phase[0] + 1j*ind_from_phase[1], step=4)
plt.show()
