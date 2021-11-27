import ltempy as lp
import numpy as np
import matplotlib.pyplot as plt
import pickle
import ncempy.io.dm as dm

# %%
file = dm.dmReader('init_center.dm3')
img = lp.SITIEImage(file)

# %%
img.data.clip_data().high_pass().shift_pos()
img.reconstruct(df=1e-3)

def test_phase():
	assert(len(img.phase.shape) == 2)

def test_B():
	assert(len(img.Bx.shape) == 2)

def test_units():
	assert(img.x_unit == 'm')
	assert(np.isclose(img.dx,2.443537348881364e-09))
