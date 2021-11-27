import ltempy as lp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def test_cielab_cmap():
	assert(type(lp.cielab_cmap()) == matplotlib.colors.ListedColormap)

X = np.linspace(0,1,100)
x, y = np.meshgrid(X, X)
data = x + 1j * y

def test_cielab_rgba():
	assert(lp.cielab_rgba(data).shape == (100,100,4))

def test_rgba():
	assert(lp.rgba(data).shape == (100,100,4))
