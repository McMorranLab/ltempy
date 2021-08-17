import ltempy as lp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

rng = np.random.default_rng()

#########################

datadir = Path("asdf/data")
outdir = Path("./outputs")
fname = Path("asdf/data/some/structure/asdf.png")
out = lp.outpath(datadir, outdir, fname)

five = rng.standard_normal(2048)
plt.title("standard normal dist")
plt.hist(five, bins = 30)
plt.show()

plt.title("clipped to 1.5 sigma")
plt.hist(lp.clip_data(five, sigma = 1.5), bins = 30)
plt.show()

plt.title("shifted positive")
plt.hist(lp.shift_pos(five), bins = 30)
plt.show()

X = np.arange(-128, 128)
x, y = np.meshgrid(X, X)
data = (np.sin(2 * np.pi * 10 / 256 * x) + np.sin(2 * np.pi * 30 / 256 * x))
data[y>0] = 0

fig, [[ax1, ax2]] = lp.subplots(12)
ax1.imshow(data.real)
ax2.ax.plot(data.real[32])
plt.show()

one = lp.high_pass(data, cutoff = 20)
fig, [[ax1, ax2]] = lp.subplots(12)
ax1.set_title("high pass, cutoff")
ax1.imshow(one.real)
ax2.ax.plot(one.real[32])
plt.show()

two = lp.low_pass(data, cutoff = 20)
fig, [[ax1, ax2]] = lp.subplots(12)
ax1.set_title("low pass, cutoff")
ax1.imshow(two.real)
ax2.ax.plot(two.real[32])
plt.show()

three = lp.high_pass(data, cutoff = 20, gaussian = True)
fig, [[ax1, ax2]] = lp.subplots(12)
ax1.set_title("high pass, gaussian")
ax1.imshow(three.real)
ax2.ax.plot(three.real[32])
plt.show()

four = lp.low_pass(data, cutoff = 20, gaussian = True)
fig, [[ax1, ax2]] = lp.subplots(12)
ax1.set_title("low pass, gaussian")
ax1.imshow(four.real)
ax2.ax.plot(four.real[32])
plt.show()

##############################
print("Starting ndap testing. ")
six = lp.ndap(rng.standard_normal((32, 32)))

plt.title("standard normal dist")
plt.hist(six.flatten(), bins = 30)
plt.show()

plt.title("clipped to 1.5 sigma")
plt.hist(six.clip_data(sigma = 1.5).flatten(), bins = 30)
plt.show()

plt.title("shifted positive")
plt.hist(six.shift_pos().flatten(), bins = 30)
plt.show()

seven = lp.ndap(data)
seven.high_pass(cutoff = 20)
fig, [[ax1, ax2]] = lp.subplots(12)
ax1.set_title("high pass, cutoff")
ax1.imshow(seven.real)
ax2.ax.plot(seven.real[32])
plt.show()

seven = lp.ndap(data)
seven.low_pass(cutoff = 20)
fig, [[ax1, ax2]] = lp.subplots(12)
ax1.set_title("low pass, cutoff")
ax1.imshow(seven.real)
ax2.ax.plot(seven.real[32])
plt.show()

seven = lp.ndap(data)
seven.high_pass(cutoff = 20, gaussian = True)
fig, [[ax1, ax2]] = lp.subplots(12)
ax1.set_title("high pass, gaussian")
ax1.imshow(seven.real)
ax2.ax.plot(seven.real[32])
plt.show()

seven = lp.ndap(data)
seven.low_pass(cutoff = 20, gaussian = True)
fig, [[ax1, ax2]] = lp.subplots(12)
ax1.set_title("low pass, gaussian")
ax1.imshow(seven.real)
ax2.ax.plot(seven.real[32])
plt.show()
