import ltempy as wt
import numpy as np
from pathlib import Path
import os, shutil

# %%
def test_high_pass():
	X = np.random.random((4,4))
	assert(wt.high_pass(X).shape == (4,4))

def test_low_pass():
	X = np.random.random((4,4))
	assert(wt.low_pass(X).shape == (4,4))

def test_clip_data():
	X = np.random.random((4,4))
	assert(wt.clip_data(X).shape == (4,4))

def test_shift_pos():
	X = np.random.random((4,4))
	assert(wt.shift_pos(X).shape == (4,4))
	assert(np.all(wt.shift_pos(X) >= 0))

def test_outpath():
	datadir = Path("./path/to/data")
	outdir = Path("./other/path")
	fname = Path("./path/to/data/and/file")
	assert(wt.outpath(datadir, outdir, fname) == Path('other/path/and/file'))
	shutil.rmtree(Path('other'))

class TestNDAP:
	def test_ndap(self):
		X = np.random.random((4,4)) + 0j
		assert(type(wt.ndap(X)) == wt.ndap)
	def test_ndap_high_pass(self):
		X = np.random.random((4,4)) + 1j
		Y = wt.ndap(X)
		Y.high_pass()
		assert(np.allclose(Y, wt.high_pass(X)))
	def test_ndap_low_pass(self):
		X = np.ones((4,4)) + 1j
		Y = wt.ndap(X)
		Y.low_pass()
		assert(np.allclose(Y, wt.low_pass(X)))
	def test_ndap_clip_data(self):
		X = np.random.random((4,4)) + 0j
		Y = wt.ndap(X)
		Y.clip_data()
		assert(np.allclose(Y, wt.clip_data(X)))
	def test_ndap_shift_pos(self):
		X = np.random.random((4,4)) + 0j
		Y = wt.ndap(X)
		Y.shift_pos()
		assert(np.allclose(Y, wt.shift_pos(X)))
