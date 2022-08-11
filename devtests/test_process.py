import ltempy as lp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

rng = np.random.default_rng()


####################
X = np.linspace(-10, 10, 512)
x, y = np.meshgrid(X, X)
data = lp.ndap(10 * np.cos(10 * y), x=X, y=X)

window = [-3, -2, 3, 4]
fig, [[ax1, ax2]] = lp.subplots(12)
ax1.set_axes(data.x, data.y)
ax1.inset(window)
ax1.imshow(data)
ax2.imshow(data.get_window(window)[2])
plt.show()


newx, newy, newdata = data.get_window(window)
print(newy.shape, newx.shape, newdata.shape)

#########################

datadir = Path("asdf/data")
outdir = Path("./outputs")
fname = Path("asdf/data/some/structure/asdf.png")
out = lp.outpath(datadir, outdir, fname)

five = rng.standard_normal(2048)
plt.title("standard normal dist")
plt.hist(five, bins=30)
plt.show()

plt.title("clipped to 1.5 sigma")
plt.hist(lp.clip_data(five, sigma=1.5), bins=30)
plt.show()

plt.title("shifted positive")
plt.hist(lp.shift_pos(five), bins=30)
plt.show()

X = np.linspace(0, 127, 128)
dx = X[1] - X[0]
x, y = np.meshgrid(X, X)
f1 = 1  # cycles per unit; f1 = 1 / 128 cycles per px
f2 = 2  # = 2 / 128 cycles per px
f1px = f1 * 1 / 128
f2px = f2 * 1 / 128
data = np.zeros_like(x)
selx = (x/8) % 2 > 1
sely = (y/8) % 2 > 1
data[selx ^ sely] = 1
data += np.sin(2 * np.pi * 3 / 128 * x)
data += np.random.random((128, 128))

blur_radius = 1  # cycles per pixel -> pixels per cycle

high = lp.high_pass(data, cutoff=5 / 128)
low = lp.low_pass(data, cutoff=0.4)
gauss = lp.gaussian_blur(data, blur_radius=blur_radius)
nopad_gauss = lp.gaussian_blur(data, blur_radius=blur_radius, padding=False)

fig, [[ax1, ax2], [ax3, ax4]] = lp.subplots(22, dpi=150)
ax1.set_title("input")
ax2.set_title("gaussian blur")
ax3.set_title("high pass")
ax4.set_title("low pass")
ax1.imshow(data)
ax2.imshow(np.real(gauss))
ax3.imshow(np.real(high))
ax4.imshow(np.real(low))
plt.show()

fig, [[ax1, ax2]] = lp.subplots(12, dpi=150)
ax1.set_title("gaussian blur")
ax2.set_title("gaussian blur no padding")
ax1.imshow(np.real(gauss))
ax2.imshow(np.real(nopad_gauss))
plt.show()

X = np.arange(-128, 128)
x, y = np.meshgrid(X, X)
data = (np.sin(2 * np.pi * 10 / 256 * x) + np.sin(2 * np.pi * 30 / 256 * x))
data[y > 0] = 0

fig, [[ax1, ax2]] = lp.subplots(12)
ax1.set_title("test data")
ax1.imshow(data.real)
ax2.ax.plot(data.real[32])
plt.show()

one = lp.high_pass(data, cutoff=20 / 256)
fig, [[ax1, ax2]] = lp.subplots(12)
ax1.set_title("high pass, cutoff = {}".format(20 / 256))
ax1.imshow(one.real)
ax2.ax.plot(one.real[32])
plt.show()

two = lp.low_pass(data, cutoff=20 / 256)
fig, [[ax1, ax2]] = lp.subplots(12)
ax1.set_title("low pass, cutoff = {}".format(20 / 256))
ax1.imshow(two.real)
ax2.ax.plot(two.real[32])
plt.show()

three = lp.high_pass(data, cutoff=20 / 256, padding=True)
fig, [[ax1, ax2]] = lp.subplots(12)
ax1.set_title("high pass, padded")
ax1.imshow(three.real)
ax2.ax.plot(three.real[32])
plt.show()

four = lp.low_pass(data, cutoff=20 / 256, padding=True)
fig, [[ax1, ax2]] = lp.subplots(12)
ax1.set_title("low pass, padded")
ax1.imshow(four.real)
ax2.ax.plot(four.real[32])
plt.show()

##############################
print("Starting ndap testing. ")
six = lp.ndap(rng.standard_normal((32, 32)))

plt.title("standard normal dist")
plt.hist(six.flatten(), bins=30)
plt.show()

plt.title("clipped to 1.5 sigma")
plt.hist(six.clip_data(sigma=1.5).flatten(), bins=30)
plt.show()

plt.title("shifted positive")
plt.hist(six.shift_pos().flatten(), bins=30)
plt.show()

seven = lp.ndap(data)
seven.high_pass(cutoff=20 / 256)
fig, [[ax1, ax2]] = lp.subplots(12)
ax1.set_title("high pass, cutoff")
ax1.imshow(seven.real)
ax2.ax.plot(seven.real[32])
plt.show()

seven = lp.ndap(data)
seven.low_pass(cutoff=20 / 256)
fig, [[ax1, ax2]] = lp.subplots(12)
ax1.set_title("low pass, cutoff")
ax1.imshow(seven.real)
ax2.ax.plot(seven.real[32])
plt.show()

seven = lp.ndap(data)
seven.gaussian_blur(blur_radius=4)
fig, [[ax1, ax2]] = lp.subplots(12)
ax1.set_title("gaussian blur")
ax1.imshow(seven.real)
ax2.ax.plot(seven.real[32])
plt.show()

seven = lp.ndap(data)
seven.high_pass(cutoff=20 / 256, padding=True)
fig, [[ax1, ax2]] = lp.subplots(12)
ax1.set_title("high pass, padded")
ax1.imshow(seven.real)
ax2.ax.plot(seven.real[32])
plt.show()

seven = lp.ndap(data)
seven.low_pass(cutoff=20 / 256, padding=True)
fig, [[ax1, ax2]] = lp.subplots(12)
ax1.set_title("low pass, padded")
ax1.imshow(seven.real)
ax2.ax.plot(seven.real[32])
plt.show()
