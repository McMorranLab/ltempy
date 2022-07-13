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

import numpy as np
from . import constants as _


def weights(M, s, s_mag, sig, z_hat, p, thickness):
    # Calculates the weights for the Mansuripur algo
    # aka, the terms in the sum on the RHS of Eq 13(a), without exp(2\pi i ...)
    Gp = G(p, sig, z_hat, thickness * s_mag)
    sig_x_z = np.cross(sig, z_hat, axisa=0, axisb=0, axisc=0)
    p_x_p_M = np.cross(p, np.cross(p, M, axisa=0, axisb=0,
                       axisc=0), axis=0, axisb=0, axisc=0)
    weights = 2 * _.e / _.hbar / _.c * 1j * thickness / s_mag * \
        Gp * np.einsum('i...,i...->...', sig_x_z, p_x_p_M)
    weights[:, 0, 0, :] = 0
    return(weights)


def laplacian_2d(data, dx, dy):
    # 2-dimensional laplacian, implemented via Fourier transform
    QX = np.fft.fftfreq(data.shape[1], dx)
    QY = np.fft.fftfreq(data.shape[0], dy)
    qx, qy = np.meshgrid(QX, QY)
    out = - 4 * _.pi**2 * (qx**2 + qy**2) * np.fft.fft2(data)
    out = np.fft.ifft2(out)
    return(out)


def inverse_laplacian_2d(data, dx, dy):
    # 2-dimensional inverse laplacian, implemented via Fourier transform
    QX = np.fft.fftfreq(data.shape[1], dx)
    QY = np.fft.fftfreq(data.shape[0], dy)
    qx, qy = np.meshgrid(QX, QY)
    out = np.nan_to_num(- np.fft.fft2(data) / 4 / _.pi **
                        2 / (qx**2 + qy**2), posinf=0, neginf=0)
    out = np.fft.ifft2(out)
    return(out)


def gradient_2d(data, dx, dy):
    # 2-dimensional gradient, implemented via Fourier transform
    QX = np.fft.fftfreq(data.shape[1], dx)
    QY = np.fft.fftfreq(data.shape[0], dy)
    qx, qy = np.meshgrid(QX, QY)
    out = np.fft.fft2(data)
    out_x = - 1j * 2 * _.pi * qx * out
    out_y = - 1j * 2 * _.pi * qy * out
    out_x = np.fft.ifft2(out_x)
    out_y = np.fft.ifft2(out_y)
    return(np.array([out_x, out_y]))


def T(qx, qy, defocus=1e-3, wavelength=1.97e-12, C_s=2.7e-3, divangle=1e-5):
    """Utility function for propagate(). Microscope transfer function.
    """
    out = aperture(qx, qy) * np.exp(-1j * chi(qx, qy, defocus, wavelength, C_s)
                                    ) * np.exp(-damping(qx, qy, defocus, wavelength, C_s, divangle))
    return(out)


def damping(qx, qy, defocus=1e-3, wavelength=1.97e-12, C_s=2.7e-3, divangle=1e-5):
    # Damping function for microscope transfer fct
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
        radius = np.min([np.max(np.abs(qx)), np.max(np.abs(qy))])
    out = np.zeros_like(qx)
    out[np.sqrt(qx**2 + qy**2) <= radius] = 1
    return(out)


def G(p, sig, z_hat, ts_mag):
    # Mansuripur, Eq 13(b)
    sum1 = np.einsum('i,i...->...', p, sig)
    sum2 = np.einsum('i,i...->...', p, z_hat)
    out = 1 / (sum1**2 + sum2**2)[np.newaxis, ...]
    out = out * np.sinc(ts_mag * sum1 / sum2)
    return(out)


def sims_shared(mx, my, mz, dx, dy):
    # Used for everything in LTEM sims
    # All outputs have shape (3, y-dim, x-dim, z-dim)
    # They just need these four dimensions so that they're broadcastable
    # mx, my, mz either 2 or 3 dim

    mx = np.atleast_3d(mx)  # (y-dim, x-dim, z-dim)
    my = np.atleast_3d(my)
    mz = np.atleast_3d(mz)
    Mx = 1 / mx.shape[0] / mx.shape[1] * np.fft.fft2(mx, axes=(0, 1))
    My = 1 / my.shape[0] / my.shape[1] * np.fft.fft2(my, axes=(0, 1))
    Mz = 1 / mz.shape[0] / mz.shape[1] * np.fft.fft2(mz, axes=(0, 1))
    M = np.array([Mx, My, Mz])  # (vec, y-dim, x-dim, z-dim)

    Sx = np.fft.fftfreq(mx.shape[1], dx)
    Sy = np.fft.fftfreq(mx.shape[0], dy)
    sx, sy = np.meshgrid(Sx, Sy)  # (y-dim, x-dim)

    # (vec, y-dim, x-dim, z-dim)
    s = np.array([sx, sy, 0 * sy])[..., np.newaxis]
    s_mag = np.sqrt(np.einsum('i...,i...->...', s, s)
                    )[np.newaxis, ...]  # (vec, y-dim, x-dim, z-dim)
    sig = s/s_mag

    # (vec, y-dim, x-dim, z-dim)
    z_hat = np.array([np.zeros_like(mx), np.zeros_like(mx), np.ones_like(mx)])
    return(M, s, s_mag, sig, z_hat)


def A_mn_components(
        xshape, yshape, zshape, selz_m, selz_z,
        selz_p, s_mag, z, sigm, sigp, sig, M, thickness, z_hat):
    # set everything to be broadcastable so we can write the A_mn equations
    # shape is: (3 vector components, y-axis, x-axis, z-axis) (thank meshgrid for switching x and y)

    A_mn = np.zeros((3, xshape, yshape, zshape), dtype=complex)
    A_mn[..., selz_m] = (2 * 1j / s_mag
                         * np.exp(2 * _.pi * s_mag * z[..., selz_m])
                         * np.sinh(_.pi * thickness * s_mag)
                         * np.cross(sigm, M, axisa=0, axisb=0, axisc=0))
    A_mn[..., selz_z] = (2 * 1j / s_mag * np.cross((
        sig
        - 0.5 * np.exp(2 * _.pi * s_mag *
                       (z[..., selz_z] - thickness / 2)) * sigm
        - 0.5 * np.exp(-2 * _.pi * s_mag *
                       (z[..., selz_z] + thickness / 2)) * sigp
    ), M, axisa=0, axisb=0, axisc=0))
    A_mn[..., selz_p] = (2 * 1j / s_mag
                         * np.exp(-2 * _.pi * s_mag * z[..., selz_p])
                         * np.sinh(_.pi * thickness * s_mag)
                         * np.cross(sigp, M, axisa=0, axisb=0, axisc=0)
                         )
    zero_comp = np.cross(
        z_hat[:, 0, 0, :], M[:, 0, 0, :], axisa=0, axisb=0, axisc=0)
    A_mn[:, 0, 0, selz_z] = -4 * _.pi * z[:, 0, 0, selz_z] * zero_comp
    A_mn[:, 0, 0, selz_m] = 2 * _.pi * thickness * zero_comp
    A_mn[:, 0, 0, selz_p] = -2 * _.pi * thickness * zero_comp
    return(A_mn)


def _extend_and_fill_mirror(data):
    ds0 = data.shape[0]
    ds1 = data.shape[1]
    bdata = np.zeros((3 * ds0, 3 * ds1), dtype=data.dtype)
    bdata[0:ds0, 0:ds1] = data[::-1, ::-1]
    bdata[0:ds0, ds1:2*ds1] = data[::-1, :]
    bdata[0:ds0, 2*ds1:3*ds1] = data[::-1, ::-1]
    bdata[ds0:2*ds0, 0:ds1] = data[:, ::-1]
    bdata[ds0:2*ds0, ds1:2*ds1] = data
    bdata[ds0:2*ds0, 2*ds1:3*ds1] = data[:, ::-1]
    bdata[2*ds0:3*ds0, 0:ds1] = data[::-1, ::-1]
    bdata[2*ds0:3*ds0, ds1:2*ds1] = data[::-1]
    bdata[2*ds0:3*ds0, 2*ds1:3*ds1] = data[::-1, ::-1]
    return(bdata)
