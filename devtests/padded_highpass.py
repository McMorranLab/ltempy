# %%
import ltempy as lp
import numpy as np
import matplotlib.pyplot as plt
import ncempy.io.ser as ser
import os
from pathlib import Path

# %%
fname = Path("/Users/wsp/Dropbox (University of Oregon)/wsp_research/data/220809_fegd3x120Pt/pos_field/img_0005_1.ser")
file = ser.serReader(fname)
img = lp.SITIEImage(file)

# %%
data = lp.shift_pos(lp.low_pass(img.data.clip_data(sigma=3), padding=False, cutoff = 1 / 128).real)

fig, [[ax]] = lp.subplots(dpi=300)
ax.imshow(data.real)
plt.show()