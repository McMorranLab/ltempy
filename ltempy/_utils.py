import numpy as np
from . import constants as _

def T(qx, qy, defocus = 1e-3, wavelength = 1.97e-12, C_s = 2.7e-3, divangle = 1e-5):
	"""Utility function for propagate(). Microscope transfer function.
	"""
	out = aperture(qx, qy) * np.exp(-1j * chi(qx, qy, defocus, wavelength, C_s)) * np.exp(-damping(qx, qy, defocus, wavelength, C_s, divangle))
	return(out)

def damping(qx, qy, defocus = 1e-3, wavelength = 1.97e-12, C_s = 2.7e-3, divangle = 1e-5):
	qp = np.sqrt(qx**2 + qy**2)
	return((_.pi * divangle / wavelength)**2 * (C_s * wavelength**3 * qp**3 + defocus * wavelength * qp)**2)

def chi(qx, qy, defocus, wavelength, C_s):
	"""Utility function for propagate(). Phase transfer function.
	"""
	return(_.pi * wavelength * defocus * (qx**2 + qy**2) + 0.5 * _.pi * C_s * wavelength**3 * (qx**2 + qy**2)**2)

def aperture(qx, qy, radius=None):
	"""Utility function for propagate(). Circular aperture.
	"""
	if radius is None:
		radius = np.max(np.sqrt(qx**2))
	out = np.zeros_like(qx)
	out[np.sqrt(qx**2 + qy**2) < radius] = 1
	return(out)

def G(p, sig, z_hat, ts_mag):
	sum1 = np.einsum('i,i...->...', p, sig)
	sum2 = np.einsum('i,i...->...', p, z_hat)
	out = 1 / (sum1**2 + sum2**2)[np.newaxis,...]
	out *= np.sinc(ts_mag * sum1 / sum2)
	return(out)

def sims_shared(mx, my, mz, dx, dy):
	#### Used for everything in LTEM sims
	#### All outputs have shape (3, y-dim, x-dim, z-dim)
	#### They just need these four dimensions so that they're broadcastable
	#### mx, my, mz either 2 or 3 dim

	mx = np.atleast_3d(mx) ### (y-dim, x-dim, z-dim)
	my = np.atleast_3d(my)
	mz = np.atleast_3d(mz)
	Mx = 1 / mx.shape[0] / mx.shape[1] * np.fft.fft2(mx, axes=(0,1))
	My = 1 / my.shape[0] / my.shape[1] * np.fft.fft2(my, axes=(0,1))
	Mz = 1 / mz.shape[0] / mz.shape[1] * np.fft.fft2(mz, axes=(0,1))
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
