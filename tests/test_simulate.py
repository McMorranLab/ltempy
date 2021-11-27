import ltempy as lp
import numpy as np
import matplotlib.pyplot as plt

# %%
X = np.linspace(-100e-9, 100e-9, 16)
Z = np.linspace(-100e-9, 100e-9, 5)
x2, y2 = np.meshgrid(X, X)
x3, y3, z3 = np.meshgrid(X, X, Z)
dx = X[1] - X[0]
dy = X[1] - X[0]

# %%
def test_jchessmodel2():
	m2 = lp.jchessmodel(x2, y2, z=0, n=1)
	assert(m2[0].shape == (16, 16))

def test_jchessmodel3():
	m3 = lp.jchessmodel(x3, y3, z3, n=1)
	assert(m3[0].shape == (16, 16, 5))

def test_ab_phase2():
	m2 = lp.jchessmodel(x2, y2, z=0, n=1)
	phase = lp.ab_phase(m2[0], m2[1], m2[2], dx, dy)
	assert(phase.shape == (16, 16))

def test_ab_phase3():
	m3 = lp.jchessmodel(x3, y3, z3, n=1)
	phase = lp.ab_phase(m3[0], m3[1], m3[2], dx, dy)
	assert(phase.shape == (16, 16, 5))

def test_B_from_mag2():
	m2 = lp.jchessmodel(x2, y2, z=0, n=1)
	Bfm2 = lp.B_from_mag(m2[0], m2[1], m2[2], z=1e-9, dx=dx, dy=dy)
	assert(Bfm2[0].shape == (16, 16))

def test_B_from_mag3():
	m2 = lp.jchessmodel(x2, y2, z=0, n=1)
	Bfm3 = lp.B_from_mag(m2[0], m2[1], m2[2], z=Z, dx=dx, dy=dy)
	assert(Bfm3[0].shape == (16, 16, 5))

def test_A_from_mag2():
	m2 = lp.jchessmodel(x2, y2, z=0, n=1)
	Afm2 = lp.A_from_mag(m2[0], m2[1], m2[2], z=1e-9, dx=dx, dy=dy)
	assert(Afm2[0].shape == (16, 16))

def test_A_from_mag3():
	m2 = lp.jchessmodel(x2, y2, z=0, n=1)
	Afm3 = lp.A_from_mag(m2[0], m2[1], m2[2], z=Z, dx=dx, dy=dy)
	assert(Afm3[0].shape == (16, 16, 5))

def test_img_from_mag2():
	m2 = lp.jchessmodel(x2, y2, z=0, n=1)
	ifm2 = lp.img_from_mag(m2[0], m2[1], m2[2], dx, dy, defocus=1e-3)
	assert(ifm2.shape == (16, 16))

def test_img_from_mag3():
	m3 = lp.jchessmodel(x3, y3, z3, n=1)
	ifm3 = lp.img_from_mag(m3[0], m3[1], m3[2], dx, dy, defocus=1e-3)
	assert(ifm3.shape == (16, 16))

def test_img_from_phase():
	m2 = lp.jchessmodel(x2, y2, z=0, n=1)
	phase = lp.ab_phase(m2[0], m2[1], m2[2], dx, dy)
	ifp = lp.img_from_phase(phase, dx, dy, defocus=1e-3)
	assert(ifp.shape == (16, 16))

def test_ind_from_phase():
	m2 = lp.jchessmodel(x2, y2, z=0, n=1)
	phase = lp.ab_phase(m2[0], m2[1], m2[2], dx, dy)
	indfp = lp.ind_from_phase(phase)
	assert(indfp[0].shape == (16, 16))

def test_sitie():
	m2 = lp.jchessmodel(x2, y2, z=0, n=1)
	ifm2 = lp.img_from_mag(m2[0], m2[1], m2[2], dx, dy, defocus=1e-3)
	pfi = lp.sitie(ifm2, defocus=1e-3, dx=dx, dy=dy)
	assert(pfi.shape == (16, 16))

def test_ind_from_mag2():
	m2 = lp.jchessmodel(x2, y2, z=0, n=1)
	Bx, By = lp.ind_from_mag(m2[0], m2[1], m2[2], dx, dy)
	assert(Bx.shape == (16, 16))

def test_ind_from_mag3():
	m3 = lp.jchessmodel(x3, y3, z3, n=1)
	Bx, By = lp.ind_from_mag(m3[0], m3[1], m3[2], dx, dy)
	assert(Bx.shape == (16, 16, 5))
