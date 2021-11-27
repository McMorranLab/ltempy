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
r"""Simulate Aharonov-Bohm phase, magnetic vector potential, magnetic field, and LTEM images.

Given a magnetic thin film with magnetization \(\mathbf{m}(x,y)\) and thickness \(\tau\), one can
come to analytical expressions [1] for the Fourier components of

1. the magnetic vector potential \(\mathbf{A}(x, y, z)\)
2. the magnetic field \(\mathbf{B}(x, y, z)\)
3. the Aharonov-Bohm phase \(\phi_m(x, y)\) acquired by an electron passing through the sample

From there, the magnetic vector potential, magnetic field, and phase can be calculated directly
using a fast fourier transform algorithm.

The Aharonov-Bohm phase can be related to the integrated perpendicular components of magnetic field via

\[
\nabla_{\perp} \phi_m = -\frac{e\tau}{\hbar} \left[\mathbf{B}\times\hat{\mathbf{e}}_{z}\right]
\]

where \(\hat{\mathbf{e}}_z\) is the direction of propagation.

Lorentz TEM images can be calculated three ways [2] (all within the framework of Fourier optics):

1. `img_from_mag`: Directly from the magnetization, using the paraxial approximation.
2. `img_from_phase`: From a phase, using the paraxial approximation.
3. `propagate`: From an exit wavefunction, without the paraxial approximation.

---

1. Mansuripur, M. Computation of electron diffraction patterns in Lorentz electron microscopy of thin magnetic films. Journal of Applied Physics 69, 2455–2464 (1991).

2. Chess, J. J. et al. Streamlined approach to mapping the magnetic induction of skyrmionic materials. Ultramicroscopy 177, 78–83 (2017).

"""

from . import constants as _
import numpy as np
from .process import shift_pos
from .sitie import ind_from_phase
from ._utils import T, sims_shared, weights, G, A_mn_components, laplacian_2d, inverse_laplacian_2d, gradient_2d

__all__ = [
			'ab_phase',
			'B_from_mag',
			'A_from_mag',
			'img_from_mag',
			'img_from_phase',
			'jchessmodel',
			'ind_from_mag',
			'propagate']

# Mansuripur
def ab_phase(mx, my, mz, dx=1, dy=1, thickness=60e-9, p = np.array([0,0,1])):
	"""Calculate the Aharonov-Bohm phase imparted on a fast electron by a magnetized sample.

	This is a direct implementation of [1], Eq (13).

	The shape of the output array is the same as mx, my, and mz. If mx, my, mz are three dimensional
	(i.e., have z-dependence as well as x and y), the third dimension of the output array represents
	the Aharonov-Bohm phase of each z-slice of the magnetization.

	**Parameters**

	* **mx** : _ndarray_ <br />
	The x-component of magnetization. Should be two or three dimensions.

	* **my** : _ndarray_ <br />
	The y-component of magnetization. Should be two or three dimensions.

	* **mz** : _ndarray_ <br />
	The z-component of magnetization. Should be two or three dimensions.

	* **dx** : _number, optional_ <br />
	The spacing between pixels/samples in mx, my, mz, in the x-direction. <br />
	Default is `dx = 1`.

	* **dy** : _number, optional_ <br />
	The spacing between pixels/samples in mx, my, mz, in the y-direction. <br />
	Default is `dy = 1`.

	* **thickness** : _number, optional_ <br />
	The thickness of each slice of the x-y plane (if no z-dependence, the thickness of the sample). <br />
	Default is `thickness = 60e-9`.

	* **p** : _ndarray, optional_ <br />
	A unit vector representing the direction of the electron's path. Shape should be `(3,)`. <br />
	Default is `p = np.array([0,0,1])`.

	**Returns**

	* _ndarray_ <br />
	The Aharonov-Bohm phase imparted on the electron by the magnetization. Shape will be the same as mx, my, mz.
	"""
	# All direct from definitions in Mansuripur
	M, s, s_mag, sig, z_hat = sims_shared(mx, my, mz, dx, dy)
	w = weights(M, s, s_mag, sig, z_hat, p, thickness)

	# multiply by weights.shape to counter ifft2's normalization
	# old versions of numpy don't have the `norm = 'backward'` option
	phase = w.shape[1] * w.shape[2] * np.fft.ifft2(w, axes=(1,2))
	return(np.squeeze(phase.real))

def B_from_mag(mx, my, mz, dx = 1, dy = 1, z = 0, thickness = 60e-9):
	r"""Calculate the magnetic field of a specified 2-d magnetic configuration.

	This is an implementation of [1], Eqn (11), which gives an analytic expression for
	the Fourier components of the magnetic vector potential. The magnetic field Fourier components
	are then calculated analytically via \(\mathbf{B} = \nabla\times\mathbf{A}\).

	The output shape is `(3, mx.shape[0], mx.shape[1], z.shape[0])`.

	**Parameters**

	* **mx** : _ndarray_ <br />
	The x-component of magnetization. Must be a 2-d array.

	* **my** : _ndarray_ <br />
	The y-component of magnetization. Must be a 2-d array.

	* **mz** : _ndarray_ <br />
	The z-component of magnetization. Must be a 2-d array.

	* **dx** : _number, optional_ <br />
	The spacing between pixels/samples in mx, my, mz, in the x-direction. <br />
	Default is `dx = 1`.

	* **dy** : _number, optional_ <br />
	The spacing between pixels/samples in mx, my, mz, in the y-direction. <br />
	Default is `dy = 1`.

	* **z** : _number, optional_, ndarray_ <br />
	The z-coordinates at which to calculate the B-field. Can be a number, or a 1d-array. <br />
	Default is `z = 0`.

	* **thickness** : _number, optional_ <br />
	The thickness of the sample. <br />
	Default is `thickness = 60e-9`.

	**Returns**

	* _ndarray_ <br />
	The magnetic field resulting from the given 2d magnetization.
	"""
	z = np.atleast_1d(z)
	selz_m = z < - thickness / 2
	selz_z = np.abs(z) <= thickness / 2
	selz_p = z > thickness / 2
	z = z[np.newaxis,np.newaxis,np.newaxis,...]

	M, s, s_mag, sig, z_hat = sims_shared(mx, my, mz, dx, dy)
	sigp = sig + 1j * z_hat
	sigm = sig - 1j * z_hat

	A_mn = A_mn_components(
			mx.shape[0], mx.shape[1], z.shape[-1], selz_m, selz_z,
			selz_p, s_mag, z, sigm, sigp, sig, M, thickness, z_hat)
	dxA_mn = 1j * 2 * _.pi * s[0][np.newaxis, ...] * A_mn
	dyA_mn = 1j * 2 * _.pi * s[1][np.newaxis, ...] * A_mn
	dzA_mn = np.zeros((3, mx.shape[0], mx.shape[1], z.shape[-1]), dtype=complex)
	dzA_mn[...,selz_m] = 2 * _.pi * s_mag * A_mn[...,selz_m]
	dzA_mn[...,selz_p] = -2 * _.pi * s_mag * A_mn[...,selz_p]
	dzA_mn[...,selz_z] = (2 * 1j / s_mag * np.cross((
						sig
						- 0.5 * 2 * _.pi * s_mag * np.exp(2 * _.pi * s_mag * (z[...,selz_z] - thickness / 2)) * sigm
						+ 0.5 * 2 * _.pi * s_mag * np.exp(-2 * _.pi * s_mag * (z[...,selz_z] + thickness / 2)) * sigp
						), M, axisa=0, axisb=0, axisc=0))
	dzA_mn[:,0,0,selz_m] = 0
	dzA_mn[:,0,0,selz_p] = 0
	dzA_mn[:,0,0,selz_z] = -4 * _.pi * np.cross(z_hat[:,0,0,:], M[:,0,0,:], axisa=0, axisb=0, axisc=0)
	B_mn = np.zeros((3, mx.shape[0], mx.shape[1], z.shape[-1]), dtype=complex)
	B_mn[0] = dyA_mn[2] - dzA_mn[1]
	B_mn[1] = dzA_mn[0] - dxA_mn[2]
	B_mn[2] = dxA_mn[1] - dyA_mn[0]
	B = B_mn.shape[1] * B_mn.shape[2] * np.fft.ifft2(B_mn, axes=(1,2))
	return(np.squeeze(B.real))

def A_from_mag(mx, my, mz, dx = 1, dy = 1, z = 0, thickness = 60e-9):
	r"""Calculate the magnetic vector potential of a specified 2-d magnetic configuration.

	This is an implementation of [1], Eqn (11), which gives the Fourier components of
	the magnetic vector potential.

	The output shape is `(3, mx.shape[0], mx.shape[1], z.shape[0])`.

	**Parameters**

	* **mx** : _ndarray_ <br />
	The x-component of magnetization. Must be a 2-d array.

	* **my** : _ndarray_ <br />
	The y-component of magnetization. Must be a 2-d array.

	* **mz** : _ndarray_ <br />
	The z-component of magnetization. Must be a 2-d array.

	* **dx** : _number, optional_ <br />
	The spacing between pixels/samples in mx, my, mz, in the x-direction. <br />
	Default is `dx = 1`.

	* **dy** : _number, optional_ <br />
	The spacing between pixels/samples in mx, my, mz, in the y-direction. <br />
	Default is `dy = 1`.

	* **z** : _number, optional_, ndarray_ <br />
	The z-coordinates at which to calculate the B-field. Can be a number_, or a 1d-array. <br />
	Default is `z = 0`.

	* **thickness** : _number, optional_ <br />
	The thickness of the sample. <br />
	Default is `thickness = 60e-9`.

	**Returns**

	* _ndarray_ <br />
	The magnetic vector potential resulting from the given 2d magnetization.
	"""
	z = np.atleast_1d(z)
	selz_m = z < -thickness/2
	selz_z = np.abs(z) <= thickness/2
	selz_p = z > thickness/2
	z = z[np.newaxis,np.newaxis,np.newaxis,...]

	M, s, s_mag, sig, z_hat = sims_shared(mx, my, mz, dx, dy)
	sigp = sig + 1j * z_hat
	sigm = sig - 1j * z_hat

	A_mn = A_mn_components(
				mx.shape[0], mx.shape[1], z.shape[-1], selz_m, selz_z,
				selz_p, s_mag, z, sigm, sigp, sig, M, thickness, z_hat)
	A = A_mn.shape[1] * A_mn.shape[2] * np.fft.ifft2(A_mn, axes=(1,2))
	return(np.squeeze(A.real))

def ind_from_mag(mx, my, mz, dx=1, dy=1, thickness=60e-9, p = np.array([0,0,1])):
	"""Calculate the integrated perpendicular magnetic field of a magnetized sample.

	This is shorthand for `ind_from_phase(ab_phase(*args))`. This method is used rather than `B_from_mag`
	because the integral over z can be done analytically.

	**Parameters**

	* **mx** : _ndarray_ <br />
	The x-component of magnetization. Should be two or three dimensions.

	* **my** : _ndarray_ <br />
	The y-component of magnetization. Should be two or three dimensions.

	* **mz** : _ndarray_ <br />
	The z-component of magnetization. Should be two or three dimensions.

	* **dx** : _number, optional_ <br />
	The spacing between pixels/samples in mx, my, mz, in the x-direction. <br />
	Default is `dx = 1`.

	* **dy** : _number, optional_ <br />
	The spacing between pixels/samples in mx, my, mz, in the y-direction. <br />
	Default is `dy = 1`.

	* **thickness** : _number, optional_ <br />
	The thickness of each slice of the x-y plane (if no z-dependence, the thickness of the sample). <br />
	Default is `thickness = 60e-9`.

	* **p** : _ndarray, optional_ <br />
	A unit vector representing the direction of the electron's path. Shape should be `(3,)`. <br />
	Default is `p = np.array([0,0,1])`.

	**Returns**

	* _ndarray_ <br />
	The x-component of the magnetic induction.

	* _ndarray_ <br />
	The y-component of the magnetic induction.
	"""
	phase = ab_phase(mx, my, mz, dx, dy, thickness, p)
	Bx, By = ind_from_phase(phase, dx, dy, thickness)
	return(np.array([Bx, By]))

def img_from_mag(mx, my, mz, dx = 1, dy = 1, defocus = 0, thickness = 60e-9, wavelength = 1.97e-12, p = np.array([0,0,1]), divangle = 1e-5):
	r"""Calculate the Lorentz TEM image from a given (2 or 3 dim) magnetization.

	This is a combination of [2], Eqn (7), which gives the output intensity in terms of \(\phi_m\) within the paraxial approximation,
	and [1], Eqn (13), which gives the Fourier components of \(\phi_m\) in terms of the magnetization.

	**Parameters**

	* **mx** : _ndarray_ <br />
	The x-component of magnetization. Should be two or three dimensions.

	* **my** : _ndarray_ <br />
	The y-component of magnetization. Should be two or three dimensions.

	* **mz** : _ndarray_ <br />
	The z-component of magnetization. Should be two or three dimensions.

	* **dx** : _number, optional_ <br />
	The spacing between pixels/samples in mx, my, mz, in the x-direction. <br />
	Default is `dx = 1`.

	* **dy** : _number, optional_ <br />
	The spacing between pixels/samples in mx, my, mz, in the y-direction. <br />
	Default is `dx = 1`.

	* **defocus** : _number, optional_ <br />
	The defocus - note that this should be non-zero in order to see any contrast. <br />
	Default is `defocus = 0`.

	* **thickness** : _number, optional_ <br />
	The thickness of each slice of the x-y plane (if no z-dependence, the thickness of the sample). <br />
	Default is `thickness = 60e-9`.

	* **wavelength** : _number, optional_ <br />
	The relativistic electron wavelength. <br />
	Default is `wavelength = 1.97e-12`.

	* **p** : _ndarray, optional_ <br />
	A unit vector representing the direction of the electron's path. Shape should be `(3,)`. <br />
	Default is `p = np.array([0,0,1])`.

	* **divangle** : _number, optional_ <br />
	The divergence angle \(\Theta_c\). <br />
	Default is `divangle = 1e-5`.

	**Returns**

	* _ndarray_ <br />
	The intensity of the image plane.
	"""
	M, s, s_mag, sig, z_hat = sims_shared(mx, my, mz, dx, dy)
	w = weights(M, s, s_mag, sig, z_hat, p, thickness)

	nabla2weights = - 4 * _.pi**2 * s_mag**2 * w
	nablaweights = 1j * 2 * _.pi * s * w

	nabla2phi = nabla2weights.shape[1] * nabla2weights.shape[2] * np.fft.ifft2(nabla2weights, axes=(1,2))
	nablaphi = nablaweights.shape[1] * nablaweights.shape[2] * np.fft.ifft2(nablaweights, axes=(1,2))
	nablaphi2 = nablaphi[0]**2 + nablaphi[1]**2

	if nablaphi2.shape[-1] > 1:
		nablaphi2 = np.sum(nablaphi2, axis=-1)
	if nabla2phi.shape[-1] > 1:
		nabla2phi = np.sum(nabla2phi, axis=-1)
	out = 1 - wavelength * defocus / 2 / _.pi * np.squeeze(nabla2phi) - (_.pi * divangle * defocus)**2 / 2 / np.log(2) * np.squeeze(nablaphi2)
	return(out.real)

def img_from_phase(phase, dx = 1, dy = 1, defocus = 0, wavelength = 1.97e-12, divangle = 1e-5):
	r"""Calculate the Lorentz TEM image given a two-dimensional phase distribution and defocus.

	This is an implementation of [2], Eqn (7), which gives the output intensity in terms of \(\phi_m\)
	within the paraxial approximation.

	**Parameters**

	* **phase** : _ndarray_ <br />
	A 2d array containing the phase of the electron at the sample plane.

	* **dx** : _number, optional_ <br />
	The spacing between pixels/samples in mx, my, mz, in the x-direction. <br />
	Default is `dx = 1`.

	* **dy** : _number, optional_ <br />
	The spacing between pixels/samples in mx, my, mz, in the y-direction. <br />
	Default is `dx = 1`.

	* **defocus** : _number, optional_ <br />
	The defocus - note that this should be non-zero in order to see any contrast. <br />
	Default is `defocus = 0`.

	* **wavelength** : _number, optional_ <br />
	The relativistic electron wavelength. <br />
	Default is `wavelength = 1.97e-12`.

	* **divangle** : _number, optional_ <br />
	The divergence angle \(\Theta_c\). <br />
	Default is `divangle = 1e-5`.

	**Returns**

	* _ndarray_ <br />
	The intensity of the image plane.
	"""
	nabla2phase = laplacian_2d(phase, dx, dy)
	nablaphase = gradient_2d(phase, dx, dy)
	nablaphase2 = nablaphase[0]**2 + nablaphase[1]**2

	img = 1 - wavelength * defocus / 2 / _.pi * nabla2phase - (_.pi * divangle * defocus)**2 / 2 / np.log(2) * nablaphase2
	return(img.real)

def propagate(mode, dx = 1, dy = 1, T=T, **kwargs):
	r"""Calculates the Lorentz image given the exit wave \(\psi_0\) and microscope transfer function \(T(\mathbf{q}_{\perp})\).

	\[\psi_f = \mathcal{F}^{-1}\left[\mathcal{F}\left[\psi_0\right] T(\mathbf{q}_{\perp}) \right]\]

	**Parameters**

	* **mode**: _complex ndarray_ <br />
	The exit wave to propagate. Dimension should be 2, may be complex or real.

	* **dx**: _number, optional_ <br />
	The x pixel size of the exit wave. <br />
	Default is `dx = 1`.

	* **dy**: _number, optional_ <br />
	The y pixel size of the exit wave. <br />
	Default is `dx = 1`.

	* **T**: _function, optional_ <br />
	The microscope transfer function. Takes **qx** and **qy** (the spatial frequencies)
	as the first two positional arguments. <br />
	The default is a common transfer function described in Ref (1), Eqns (4-6), that takes the following kwargs: <br />
		<ul>
			<li> **defocus** : _number, optional_ <br />
			Default is `defocus = 1e-3`.
			</li>
			<li> **wavelength** : _number, optional_ <br />
			Default is `wavelength = 1.97e-12` (for 300keV electrons).
			</li>
			<li> **C_s** : _number, optional_ <br />
			The spherical aberration coefficient of the microscope. <br />
			Default is `C_s = 2.7e-3`.
			</li>
			<li> **divangle** : _number, optional_ <br />
			The divergence angle. <br />
			Default is `divangle = 1e-5`.
		</ul>

	* ****kwargs**: _optional_ <br />
	Extra arguments to be passed to the transfer function.

	**Returns**

	* _complex ndarray_ <br />
	The transverse complex amplitude in the image plane. Output has the same
	shape as x, y, and mode.
	"""
	U = np.fft.fftfreq(mode.shape[1], dx)
	V = np.fft.fftfreq(mode.shape[0], dy)
	qx, qy = np.meshgrid(U, V)
	psi_q = np.fft.fft2(mode)
	psi_out = np.fft.ifft2(psi_q * T(qx, qy, **kwargs))
	return(psi_out)

# Miscellaneous
def jchessmodel(x, y, z=0, **kwargs):
	"""Calculates the magnetization of a hopfion based on Jordan Chess' model.

	**Parameters**

	* **x** : _number, ndarray_ <br />
	The x-coordinates over which to calculate magnetization.

	* **y** : _number, ndarray_ <br />
	The y-coordinates over which to calculate magnetization.

	* **z** : _number, ndarray, optional_ <br />
	The z-coordinates over which to calculate magnetization. Note, if z is an
	ndarray, then x, y, and z should have the same shape rather than relying
	on array broadcasting. <br />
	Default is `z = 0`.

	* **aa**, **ba**, **ca** : _number, optional_ <br />
	In this model, the thickness of the domain wall is set by a
	Gaussian function, defined as `aa * exp(-ba * z**2) + ca`. <br />
	Defaults are `aa = 5`, `ba = 5`, `ca = 0`.

	* **ak**, **bk**, **ck** : _number, optional_ <br />
	In this model, the thickness of the core is set by a Gaussian function,
	defined as `ak * exp(-bk * z**2) + ck`. <br />
	Defaults are `ak = 5e7`, `bk = -50`, `ck = 0`.

	* **bg**, **cg** : _number, optional_ <br />
	In this model, the helicity varies as a function of z, given
	as `pi / 2 * tanh( bg * z ) + cg`. <br />
	Defaults are `bg = 5e7`, `cg = pi/2`.

	* **n** : _number, optional_ <br />
	The skyrmion number. <br />
	Default is `n = 1`.

	**Returns**

	* _ndarray_ <br />
	The x-component of magnetization. Shape will be the same as x and y.

	* _ndarray_ <br />
	The y-component of magnetization. Shape will be the same as x and y.

	* _ndarray_ <br />
	The z-component of magnetization. Shape will be the same as x and y.
	"""
	p = {   'aa':5, 'ba':5, 'ca':0,
			'ak':5e7, 'bk':-5e1, 'ck':0,
			'bg':5e7, 'cg':_.pi/2, 'n': 1}
	for key in kwargs.keys():
		if not key in p.keys(): return("Error: {:} is not a kwarg.".format(key))
	p.update(kwargs)

	r, phi = np.sqrt(x**2+y**2), np.arctan2(y,x)

	alpha_z = p['aa'] * np.exp(-p['ba'] * z**2) + p['ca']
	k_z = p['ak'] * np.exp(-p['bk'] * z**2) + p['ck']
	gamma_z = _.pi / 2 * np.tanh(p['bg'] * z) + p['cg']
	Theta_rz = 2 * np.arctan2((k_z * r)**alpha_z, 1)

	mx = np.cos(p['n']*phi%(2*_.pi)-gamma_z) * np.sin(Theta_rz)
	my = np.sin(p['n']*phi%(2*_.pi)-gamma_z) * np.sin(Theta_rz)
	mz = np.cos(Theta_rz)
	return(np.array([mx, my, mz]))
