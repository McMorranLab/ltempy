import ltempy as wt
from ltempy import np, plt
import matplotlib

def test_cielab_cmap():
	assert(type(wt.cielab_cmap()) == matplotlib.colors.ListedColormap)

X = np.linspace(0,1,100)
x, y = np.meshgrid(X, X)
data = x + 1j * y

def test_cielab_image():
	assert(wt.cielab_image(data).shape == (100,100,4))

def test_rgba():
	assert(wt.rgba(data).shape == (100,100,4))
