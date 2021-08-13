import ltempy as wt
import matplotlib.pyplot as plt
import numpy as np
import ncempy.io.dm as dm
from pathlib import Path
import os

# %%
fpath = Path('../tests/init_center.dm3')
# fpath = Path(os.environ['wpr']).joinpath(
# 'data/201202_wsp_lorentz_holey_SiN/raw/pos_to_neg/n60.dm3'
# )
file = dm.dmReader(fpath)
img = wt.lorentz(file)

img.data.clip_data().high_pass().shift_pos()
img.sitie(1e-3)

# %%
window = (2, 2.5, 2, 2.75)
fig, [[ax1, ax2]] = wt.subplots(12, dpi=150)
ax1.origin = 'upper'
ax1.set_axes(1e6*img.x, 1e6*img.y)
ax1.inset(window)
ax1.imshow(img.data)
ax2.origin = ax1.origin
ax2.set_axes(ax1.x, ax1.y)
ax2.set_window(window)
ax2.rgba(img.Bx + 1j*img.By, cmap='viridis', brightness='amplitude')
ax2.quiver(img.Bx + 1j*img.By, step=12, color='white')
ax2.colorwheel(cmap='viridis', brightness='amplitude')
plt.show()

# %%
fig, [[ax1, ax2]] = wt.subplots(12, dpi=150)
ax1.origin = 'lower'
ax2.origin = ax1.origin
ax1.set_axes(1e6*img.x, 1e6*img.y)
ax2.set_axes(ax1.x, ax1.y)

ax1.inset(window)
ax1.imshow(img.data)

ax2.set_window(window)
ax2.rgba(img.Bx + 1j*img.By, brightness='amplitude')
ax2.quiver(img.Bx + 1j*img.By, step=12, color='white')
ax2.colorwheel(brightness='amplitude')
plt.show()
