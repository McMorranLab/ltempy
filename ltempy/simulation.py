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

from . import constants as _
import numpy as np
from .image_processing import shift_pos
from .sitie import ind_from_phase

__all__ = [
			'ab_phase',
			'B_from_mag',
			'A_from_mag',
			'img_from_mag',
			'img_from_phase',
			'jchessmodel',
			'ind_from_mag',
			'propagate']

def ind_from_mag(mx, my, mz, dx=1, dy=1, thickness=60e-9, p = np.array([0,0,1])):
	"""Calculate the magnetic induction imparted on a fast electron by a magnetized sample.

	This is shorthand for `ind_from_phase(ab_phase(*args))`.

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
	A unit vector representing the direction of the electron's path. Shape should be (3,). <br />
	Default is `p = np.array([0,0,1])`.

	**Returns**

	* **Bx** : _ndarray_ <br />
	The x-component of the magnetic induction.

	* **By** : _ndarray_ <br />
	The y-component of the magnetic induction.
	"""
	phase = ab_phase(mx, my, mz, dx, dy, thickness, p)
	if len(phase.shape) > 2:
		phase = np.sum(phase, axis=-1)
	Bx, By = ind_from_phase(phase, thickness)
	return(np.array([Bx, By]))

def ab_phase(mx, my, mz, dx=1, dy=1, thickness=60e-9, p = np.array([0,0,1])):
	"""Calculate the Aharonov-Bohm phase imparted on a fast electron by a magnetized sample.

	This is an implementation of the algorithm described in Mansuripur et. al, _Computation of electron diffraction patterns in Lorentz TEM_, Eq (13) specifically.

	The shape of the output array is the same as mx, my, and mz. If mx, my, mz are three dimensional (i.e., have z-dependence as well as x and y), the third dimension of the output array represents the Aharonov-Bohm phase of each z-slice of the magnetization.

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
	A unit vector representing the direction of the electron's path. Shape should be (3,). <br />
	Default is `p = np.array([0,0,1])`.

	**Returns**

	* **phase** : _ndarray_ <br />
	The Aharonov-Bohm phase imparted on the electron by the magnetization. Shape will be the same as mx, my, mz.
	"""
	M, s, s_mag, sig, z_hat = common_stuff(mx, my, mz, dx, dy)
	Gp = G(p, sig, z_hat, thickness * s_mag)
	sig_x_z = np.cross(sig, z_hat, axisa=0, axisb=0, axisc=0)
	p_x_p_M = np.cross(p, np.cross(p, M, axisa=0, axisb=0, axisc=0), axis=0, axisb=0, axisc=0)
	weights = 2 * _.e / _.hbar / _.c * 1j * thickness / s_mag * Gp * np.einsum('i...,i...->...',sig_x_z, p_x_p_M)
	weights[:,0,0,:] = 0
	phase = weights.shape[1] * weights.shape[2] * np.fft.ifft2(weights, axes=(1,2))
	return(np.squeeze(phase.real))

def B_from_mag(mx, my, mz, z = 0, dx = 1, dy = 1, thickness = 60e-9):
	"""Calculate the magnetic field of a specified 2-d magnetic configuration.

	This is an adaptation of the algorithm described in Mansuripur et. al, _Computation of electron diffraction patterns in Lorentz TEM_, Eq (13) specifically.

	The output shape is (3, mx.shape[0], mx.shape[1], z.shape[0]).

	**Parameters**

	* **mx** : _ndarray_ <br />
	The x-component of magnetization. Must be a 2-d array.

	* **my** : _ndarray_ <br />
	The y-component of magnetization. Must be a 2-d array.

	* **mz** : _ndarray_ <br />
	The z-component of magnetization. Must be a 2-d array.

	* **z** : _number, optional_, ndarray_ <br />
	The z-coordinates at which to calculate the B-field. Can be a number, or a 1d-array. <br />
	Default is `z = 0`.

	* **dx** : _number, optional_ <br />
	The spacing between pixels/samples in mx, my, mz, in the x-direction. <br />
	Default is `dx = 1`.

	* **dy** : _number, optional_ <br />
	The spacing between pixels/samples in mx, my, mz, in the y-direction. <br />
	Default is `dy = 1`.

	* **thickness** : _number, optional_ <br />
	The thickness of the sample. <br />
	Default is `thickness = 60e-9`.

	**Returns**

	* **B** : _ndarray_ <br />
	The magnetic field resulting from the given 2d magnetization.
	"""
	z = np.atleast_1d(z)
	selz_m = z < - thickness / 2
	selz_z = np.abs(z) <= thickness / 2
	selz_p = z > thickness / 2
	z = z[np.newaxis,np.newaxis,np.newaxis,...]

	M, s, s_mag, sig, z_hat = common_stuff(mx, my, mz, dx, dy)
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

def A_from_mag(mx, my, mz, z = 0, dx = 1, dy = 1, thickness = 60e-9):
	"""Calculate the magnetic vector potential of a specified 2-d magnetic configuration.

	This is an adaptation of the algorithm described in Mansuripur et. al, _Computation of electron diffraction patterns in Lorentz TEM_, Eq (13) specifically.

	The output shape is (3, mx.shape[0], mx.shape[1], z.shape[0]).

	**Parameters**

	* **mx** : _ndarray_ <br />
	The x-component of magnetization. Must be a 2-d array.

	* **my** : _ndarray_ <br />
	The y-component of magnetization. Must be a 2-d array.

	* **mz** : _ndarray_ <br />
	The z-component of magnetization. Must be a 2-d array.

	* **z** : _number, optional_, ndarray_ <br />
	The z-coordinates at which to calculate the B-field. Can be a number_, or a 1d-array. <br />
	Default is `z = 0`.

	* **dx** : _number, optional_ <br />
	The spacing between pixels/samples in mx, my, mz, in the x-direction. <br />
	Default is `dx = 1`.

	* **dy** : _number, optional_ <br />
	The spacing between pixels/samples in mx, my, mz, in the y-direction. <br />
	Default is `dy = 1`.

	* **thickness** : _number, optional_ <br />
	The thickness of the sample. <br />
	Default is `thickness = 60e-9`.

	**Returns**

	* **B** : _ndarray_ <br />
	The magnetic vector potential resulting from the given 2d magnetization.
	"""
	z = np.atleast_1d(z)
	selz_m = z < -thickness/2
	selz_z = np.abs(z) <= thickness/2
	selz_p = z > thickness/2
	z = z[np.newaxis,np.newaxis,np.newaxis,...]

	M, s, s_mag, sig, z_hat = common_stuff(mx, my, mz, dx, dy)
	sigp = sig + 1j * z_hat
	sigm = sig - 1j * z_hat

	A_mn = A_mn_components(
				mx.shape[0], mx.shape[1], z.shape[-1], selz_m, selz_z,
				selz_p, s_mag, z, sigm, sigp, sig, M, thickness, z_hat)
	A = A_mn.shape[1] * A_mn.shape[2] * np.fft.ifft2(A_mn, axes=(1,2))
	return(np.squeeze(A.real))

def img_from_mag(mx, my, mz, dx = 1, dy = 1, defocus = 0, thickness = 60e-9, wavelength = 1.97e-12, p = np.array([0,0,1])):
	"""Calculate the theoretical Lorentz TEM image from a given (2 or 3 dim) magnetization.

	This is an implementation of the SITIE equation (Eq 10) from J. Chess et al. 2017, _Streamlined approach to mapping the magnetic induction of skyrmionic materials_, in combination with Mansuripur et al.

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

	* **wavelength** : _number, optional_ <br />
	The relativistic electron wavelength. <br />
	Default is `wavelength = 1.97e-12`.

	**Returns**

	* **img** : _ndarray_ <br />
	The intensity of the image plane.
	"""
	M, s, s_mag, sig, z_hat = common_stuff(mx, my, mz, dx, dy)
	Gp = G(p, sig, z_hat, thickness * s_mag)
	sig_x_z = np.cross(sig, z_hat, axisa=0, axisb=0, axisc=0)
	p_x_p_M = np.cross(p, np.cross(p, M, axisa=0, axisb=0, axisc=0), axis=0, axisb=0, axisc=0)
	### weights is the only thing different from ab_phase (\nabla^2 = - 4 pi^2 s_mag^2)
	weights = - 4 * _.pi **2 * 2 * _.e / _.hbar / _.c * 1j * thickness * s_mag * Gp * np.einsum('i...,i...->...',sig_x_z, p_x_p_M)
	weights[:,0,0,:] = 0

	out = weights.shape[1] * weights.shape[2] * wavelength * defocus / 2 / _.pi * np.fft.ifft2(weights, axes=(1,2))
	if out.shape[-1] > 1:
		out = np.sum(out, axis=-1)
	out = 1 - np.squeeze(out)
	return(out.real)

def img_from_phase(phase, dx = 1, dy = 1, defocus = 0, wavelength = 1.97e-12):
	"""Calculate the Lorentz TEM image given a two-dimensional phase distribution and defocus.

	This is an implementation of the SITIE equation (Eq 10) from J. Chess et al. 2017, _Streamlined approach to mapping the magnetic induction of skyrmionic materials_.

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

	**Returns**

	* **img** : _ndarray_ <br />
	The intensity of the image plane.
	"""
	Sx = np.fft.fftfreq(phase.shape[1], dx)
	Sy = np.fft.fftfreq(phase.shape[0], dy)
	sx, sy = np.meshgrid(Sx, Sy) ### (y-dim, x-dim, z-dim)
	phase = - 4 * _.pi**2 * (sx**2 + sy**2) * np.fft.fft2(phase)
	img = 1 - wavelength * defocus / 2 / _.pi * np.fft.ifft2(phase)
	return(img.real)

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

	* **mx** : _ndarray_ <br />
	The x-component of magnetization. Shape will be the same as x and y.

	* **my** : _ndarray_ <br />
	The y-component of magnetization. Shape will be the same as x and y.

	* **mz** : _ndarray_ <br />
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

def T(qx, qy, defocus = 1e-3, wavelength = 1.97e-12):
	"""Utility function for propagate(). Microscope transfer function.
	"""
	out = aperture(qx, qy) * np.exp(-1j * chi(qx, qy, defocus, wavelength))
	return(out)

def propagate(mode, dx = 1, dy = 1, T=T, **targs):
	"""Calculates the Lorentz image given the exit wave, defocus, and microscope transfer function.

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
	Default takes into account defocus and an aperture.

	* ****targs**: _optional_ <br />
	Extra arguments to be passed to the transfer function.

	**Returns**

	* **psi_out** : _complex ndarray_ <br />
	The transverse complex amplitude in the image plane. Output has the same
	shape as x, y, and cphase.
	"""
	U = np.fft.fftfreq(cphase.shape[1], dx)
	V = np.fft.fftfreq(cphase.shape[0], dy)
	qx, qy = np.meshgrid(U, V)
	psi_q = np.fft.fft2(mode)
	psi_out = np.fft.ifft2(psi_q * T(qx, qy, **targs))
	return(psi_out)


def chi(qx, qy, defocus, wavelength):
	"""Utility function for propagate(). Phase transfer function.
	"""
	return(_.pi * wavelength * defocus * (qx**2 + qy**2))

def aperture(qx, qy):
	"""Utility function for propagate(). Circular aperture.
	"""
	out = np.zeros_like(qx)
	out[np.sqrt(qx**2 + qy**2) < np.max(np.sqrt(qx**2 + qy**2))] = 1
	return(out)


######## Helper functions

def G(p, sig, z_hat, ts_mag):
	sum1 = np.einsum('i,i...->...', p, sig)
	sum2 = np.einsum('i,i...->...', p, z_hat)
	out = 1 / (sum1**2 + sum2**2)[np.newaxis,...]
	out *= np.sinc(ts_mag * sum1 / sum2)
	return(out)

def common_stuff(mx, my, mz, dx, dy):
	#### Used for everything in LTEM sims
	#### All outputs have shape (3, y-dim, x-dim, z-dim)
	#### They just need these four dimensions so that they're broadcastable
	#### mx, my, mz either 2 or 3 dim

	mx = np.atleast_3d(mx) ### (y-dim, x-dim, z-dim)
	my = np.atleast_3d(my)
	mz = np.atleast_3d(mz)
	Mx = mx.shape[0] * mx.shape[1] * np.fft.fft2(mx, axes=(0,1))
	My = my.shape[0] * my.shape[1] * np.fft.fft2(my, axes=(0,1))
	Mz = mz.shape[0] * mz.shape[1] * np.fft.fft2(mz, axes=(0,1))
	M = np.array([Mx, My, Mz]) ### (vec, y-dim, x-dim, z-dim)

	Sx = np.fft.fftfreq(mx.shape[1], dx)
	Sy = np.fft.fftfreq(mx.shape[0], dy)
	sx, sy = np.meshgrid(Sx, Sy) ### (y-dim, x-dim)

	s = np.array([sx, sy, 0*sy])[...,np.newaxis] ### (vec, y-dim, x-dim, z-dim)
	s_mag = np.sqrt(np.einsum('i...,i...->...',s,s))[np.newaxis,...] ### (vec, y-dim, x-dim, z-dim)
	sig = s/s_mag

	z_hat = np.array([np.zeros_like(mx), np.zeros_like(mx), np.ones_like(mx)]) ### (vec, y-dim, x-dim, z-dim)
	return(M, s, s_mag, sig, z_hat)

def A_mn_components(
		xshape, yshape, zshape, selz_m, selz_z,
		selz_p, s_mag, z, sigm, sigp, sig, M, thickness, z_hat):
	### set everything to be broadcastable so we can write the A_mn equations
	### shape is: (3 vector components, y-axis, x-axis, z-axis) (thank meshgrid for switching x and y)

	A_mn = np.zeros((3, xshape, yshape, zshape), dtype=complex)
	A_mn[...,selz_m] = (2 * 1j / s_mag
						* np.exp(2 * _.pi * s_mag * z[...,selz_m])
						* np.sinh(_.pi * thickness * s_mag)
						* np.cross(sigm, M, axisa=0, axisb=0, axisc=0))
	A_mn[...,selz_z] = (2 * 1j / s_mag * np.cross((
						sig
						- 0.5 * np.exp(2 * _.pi * s_mag * (z[...,selz_z] - thickness / 2)) * sigm
						- 0.5 * np.exp(-2 * _.pi * s_mag * (z[...,selz_z] + thickness / 2)) * sigp
						), M, axisa=0, axisb=0, axisc=0))
	A_mn[...,selz_p] = (2 * 1j / s_mag
						* np.exp(-2 * _.pi * s_mag * z[...,selz_p])
						* np.sinh(_.pi * thickness * s_mag)
						* np.cross(sigp, M, axisa=0, axisb=0, axisc=0)
						)
	zero_comp = np.cross(z_hat[:,0,0,:], M[:,0,0,:], axisa=0, axisb=0, axisc=0)
	A_mn[:,0,0,selz_z] = -4 * _.pi * z[:,0,0,selz_z] * zero_comp
	A_mn[:,0,0,selz_m] = 2 * _.pi * thickness * zero_comp
	A_mn[:,0,0,selz_p] = -2 * _.pi * thickness * zero_comp
	return(A_mn)
