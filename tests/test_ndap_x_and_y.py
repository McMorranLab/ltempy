# %%
from tkinter import E
import ltempy as lp
import numpy as np
import pytest

# %%
X = 2 * np.arange(10)
Y = 2 * np.arange(15)
x, y = np.meshgrid(X, Y)
data = np.ones_like(x)

# %%
def test_base_case():
  test = lp.ndap(data)
  x_should_be = np.arange(10)
  y_should_be = np.arange(15)
  assert(test.dx == 1)
  assert(test.dy == 1)
  assert(np.all(test.x == x_should_be))
  assert(np.all(test.y == y_should_be))

def test_both_x_and_y():
  test = lp.ndap(data, X, Y)
  assert(test.dx == 2)
  assert(test.dy == 2)

def test_x_not_ndarray():
  with pytest.raises(Exception) as info:
    test = lp.ndap(data, "asdf", Y)
  assert "numpy.ndarray" in str(info.value)

def test_x_not_right_shape():
  with pytest.raises(Exception) as info:
    test = lp.ndap(data, x, Y)
  assert "match" in str(info.value)
  assert "shape" in str(info.value)

def test_attribute_persistance_squaring():
  test = lp.ndap(data, X, Y)
  test = test**2
  assert test.dx == 2

def test_attribute_persistance_high_pass():
  test = lp.ndap(data, X, Y)
  test.high_pass()
  assert test.dx == 2

def test_attribute_persistance_slicing():
  test = lp.ndap(data, X, Y)
  assert test[:5, :3] is not None