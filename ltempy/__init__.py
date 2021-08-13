# ltempy is a set of LTEM analysis and simulation tools developed by WSP as a member of the McMorran Lab
# Copyright (C) 2021  William S. Parker
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
ltempy contains tools for Lorentz TEM data analysis, simulation, and presentation.

Features:

* Single Image Transport of Intensity Equation (SITIE) reconstruction
* simulations - calculations of phase, B, A, image
* basic image processing - high_pass, low_pass, clipping
* a matplotlib.pyplot wrapper tailored to presenting induction maps and Lorentz data
* an implementation of the CIELAB colorspace
* module-wide unit scaling (i.e., working in nm rather than m)
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

from .cielab import *
from .constants import set_units
from .image_processing import *
from .pyplotwrapper import *
from .sitie import *
from .simulation import *
