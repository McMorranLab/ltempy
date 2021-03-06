import ltempy as lp
import numpy as np
from pathlib import Path

# %%


def test_high_pass():
    X = np.random.random((4, 4))
    assert(lp.high_pass(X).shape == (4, 4))


def test_low_pass():
    X = np.random.random((4, 4))
    assert(lp.low_pass(X).shape == (4, 4))


def test_gaussian_blur():
    X = np.random.random((4, 4))
    assert(lp.gaussian_blur(X).shape == (4, 4))
    assert(lp.gaussian_blur(X, padding=False).shape == (4, 4))


def test_clip_data():
    X = np.random.random((4, 4))
    assert(lp.clip_data(X).shape == (4, 4))


def test_shift_pos():
    X = np.random.random((4, 4))
    assert(lp.shift_pos(X).shape == (4, 4))
    assert(np.all(lp.shift_pos(X) >= 0))


def test_outpath():
    datadir = Path("./path/to/data")
    outdir = Path("./other/path")
    fname = Path("./path/to/data/and/file")
    assert(lp.outpath(datadir, outdir, fname) == Path('other/path/and/file'))


class TestNDAP:
    def test_ndap(self):
        X = np.random.random((4, 4)) + 0j
        assert(type(lp.ndap(X)) == lp.ndap)

    def test_ndap_high_pass(self):
        X = np.random.random((4, 4)) + 1j
        Y = lp.ndap(X)
        Y.high_pass()
        assert(np.allclose(Y, lp.high_pass(X)))

    def test_ndap_low_pass(self):
        X = np.ones((4, 4)) + 1j
        Y = lp.ndap(X)
        Y.low_pass()
        assert(np.allclose(Y, lp.low_pass(X)))

    def test_ndap_gaussian_blur(self):
        X = np.ones((4, 4)) + 1j
        Y = lp.ndap(X)
        Y.gaussian_blur()
        assert(np.allclose(Y, lp.gaussian_blur(X)))

    def test_ndap_clip_data(self):
        X = np.random.random((4, 4)) + 0j
        Y = lp.ndap(X)
        Y.clip_data()
        assert(np.allclose(Y, lp.clip_data(X)))

    def test_ndap_shift_pos(self):
        X = np.random.random((4, 4))
        Y = lp.ndap(X)
        Y.shift_pos()
        assert(np.allclose(Y, lp.shift_pos(X)))
