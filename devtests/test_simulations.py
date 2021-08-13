import ltempy as wt
from ltempy import np, plt

X = 1e-9 * np.linspace(-100, 100, 128)
Y = 1e-9 * np.linspace(-100, 100, 129)
Z = 1e-9 * np.linspace(-100, 100, 11)
x2, y2 = np.meshgrid(X, Y)
x3, y3, z3 = np.meshgrid(X, Y, Z)
dx = X[1] - X[0]
dy = Y[1] - Y[0]

# %%
m2 = wt.jchessmodel(x2, y2, z=0, n=1)
plt.imshow(wt.rgba(m2[0]+1j*m2[1]))
plt.title("2d mag")
plt.show()

# %%
m3 = wt.jchessmodel(x3, y3, z3, n=1)
plt.imshow(wt.rgba(m3[0][:,:,0]+1j*m3[1][:,:,0]))
plt.title("3d mag (slice)")
plt.show()

# %%
abp2 = wt.ab_phase(m2[0], m2[1], m2[2], dx, dy)
plt.imshow(abp2)
plt.title("ab phase 2d")
plt.colorbar()
plt.show()

# %%
abp3 = wt.ab_phase(m3[0], m3[1], m3[2], dx, dy)
plt.imshow(np.sum(abp3, axis=-1))
plt.title("ab phase 3d (slice)")
plt.show()

# %%
Bfm2 = wt.B_from_mag(m2[0], m2[1], m2[2], z=1e-9, dx=dx, dy=dy)
plt.imshow(wt.rgba(Bfm2[0] + 1j * Bfm2[1]))
plt.title("B from mag 2d")
plt.show()

# %%
Bfm3 = wt.B_from_mag(m2[0], m2[1], m2[2], z=Z, dx=dx, dy=dy)
plt.imshow(wt.rgba(np.sum(Bfm3[0], axis=-1) + 1j * np.sum(Bfm3[1], axis=-1)))
plt.title("B from mag 3d (sum)")
plt.show()

# %%
Afm2 = wt.A_from_mag(m2[0], m2[1], m2[2], z=0, dx=dx, dy=dy)
plt.imshow(wt.rgba(Afm2[0]+1j*Afm2[1]))
plt.title("A from mag 2d")
plt.show()

# %%
Afm3 = wt.A_from_mag(m2[0], m2[1], m2[2], z=Z, dx=dx, dy=dy)
plt.imshow(wt.rgba(np.sum(Afm3[0], axis=-1) + 1j * np.sum(Afm3[1], axis=-1)))
plt.title("A from mag 3d (sum)")
plt.show()

# %%
ifm2 = wt.img_from_mag(m2[0], m2[1], m2[2], dx, dy, defocus=1e-3)
plt.imshow(ifm2)
plt.title("img from mag 2d")
plt.colorbar()
plt.show()

# %%
ifm3 = wt.img_from_mag(m3[0], m3[1], m3[2], dx, dy, defocus=1e-3)
plt.imshow(ifm3)
plt.title("img from mag 3d (slice)")
plt.colorbar()
plt.show()

# %%
ifp = wt.img_from_phase(abp2, dx, dy, defocus=1e-3)
plt.imshow(ifp)
plt.title("img from phase")
plt.colorbar()
plt.show()

# %%
indfp = wt.ind_from_phase(abp2)
plt.imshow(wt.rgba(indfp[0] + 1j*indfp[1]))
plt.title("induction from phase")
plt.show()

# %%
pfi = wt.phase_from_img(ifm2, defocus=1e-3, dx=dx, dy=dy)
plt.imshow(pfi)
plt.title("phase from image")
plt.colorbar()
plt.show()

# %%
ifi = wt.ind_from_img(ifm2, defocus=1e-3, dx=dx, dy=dy)
plt.imshow(wt.rgba(ifi[0] + 1j*ifi[1]))
plt.title("induction from image")
plt.show()

# %%
ifm = wt.ind_from_mag(m2[0], m2[1], m2[2], dx=dx, dy=dy)
plt.imshow(wt.rgba(ifm[0] + 1j*ifm[1]))
plt.title("induction from mag (2)")
plt.show()

# %%
ifm = wt.ind_from_mag(m3[0], m3[1], m3[2], dx=dx, dy=dy)
plt.imshow(wt.rgba(ifm[0]+1j*ifm[1]))
plt.title("induction from mag (3)")
plt.show()
