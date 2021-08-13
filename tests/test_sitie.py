import ltempy as wt
from ltempy import plt, np
import pickle
import ncempy.io.dm as dm

# %%
file = dm.dmReader('init_center.dm3')
img = wt.lorentz(file)

# %%
file2 = dm.dmReader('TimeSeriesImages_10images05.dm3')
imgs = wt.lorentz(file2)
img2 = imgs[0]

# %%
img.data.clip_data().high_pass().shift_pos()
img.sitie(df=1e-3)

def test_phase():
	assert(len(img.phase.shape) == 2)

def test_B():
	assert(len(img.Bx.shape) == 2)

def test_units():
	assert(img.xUnit == 'm')
	assert(np.isclose(img.dx,2.443537348881364e-09))

# %%
img2.data.clip_data().high_pass().shift_pos()
img2.sitie(df=1e-3)

def test_phase():
	assert(len(img2.phase.shape) == 2)

def test_B():
	assert(len(img2.Bx.shape) == 2)

def test_units():
	assert(img2.xUnit == 'm')
	assert(np.isclose(img2.dx,1.32e-08))
