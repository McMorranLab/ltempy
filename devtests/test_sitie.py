import ltempy as wt
from ltempy import plt, np
import pickle

# %%
def load_data():
	with open("001_parallel.pkl", "rb") as f:
		return(pickle.load(f))

file = load_data()
file['data'] = np.array(file['data'])
file['coords'] = np.array(file['coords'])

img = wt.lorentz(file)

# %%
img.data.clip_data().high_pass().shift_pos()
img.sitie(df=1e-3)

# %%
plt.imshow(img.phase)
plt.colorbar()
plt.show()

# %%
fig, [[ax]] = wt.subplots()
ax.set_window(window=(40,60,40,60))
ax.set_xytitle("x ({})".format(img.dx))
ax.rgba(img.Bx + 1j*img.By)
ax.quiver(img.Bx + 1j*img.By, step=8, color='white')
plt.show()
