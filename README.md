# ltempy
ltempy is a set of tools for Lorentz TEM data analysis, simulation, and presentation.

# Features

* Single Image Transport of Intensity Equation (SITIE) reconstruction
* simulations - calculations of phase, B, A, image
* basic image processing - high_pass, low_pass, clipping
* a matplotlib.pyplot wrapper tailored to presenting induction maps and Lorentz data
* an implementation of the CIELAB colorspace
* module-wide unit scaling (i.e., working in nm rather than m)

## Installation

```Bash
python -m pip install ltempy
```

## Documentation

Documentation is available at [https://mcmorranlab.github.io/ltempy/](https://mcmorranlab.github.io/ltempy/).

## Tests

Tests are split into two subdirectories:

1. `tests`
	These are typical unit tests, that assert that functions return the right shape, beam parameters return the right values, etc. Run with `pytest`.
2. `devtests`
	These are tests of the actual functionality, that require a trained eye to evaluate. For example, a function `test_bessel()` will generate a bessel beam using `ltempy.bessel()`, but rather than asserting a unit test, will just plot the bessel beam so the developer can manually assert whether it looks correct. Run as normal `.py` scripts.

The rationale for `devtests` is that this package is math-heavy, so it's highly possible for the code to run fine, but be wrong. The easiest way to test for this is to check base cases where the developer knows what to look for.
