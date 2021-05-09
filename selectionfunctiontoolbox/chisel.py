#!/usr/bin/env python
#
# chisel.py
# Uses spherical needlets to estimate a selection function across the sky.
#
# Copyright (C) 2021  Douglas Boubert & Andrew Everall
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#

import numpy as np
import healpy as hp
import tqdm
import h5py
from .base import Base

class Chisel(Base):

    basis_keyword = 'wavelet'

    def _process_basis_options(self, needlet = 'littlewoodpaley', j=[0], B = 2.0, p = 1.0, wavelet_tol = 1e-10):

        if type(j) in [list,tuple,np.ndarray]:
            self.j = sorted([int(_j) for _j in j])
        else:
            self.j = [_j for _j in range(-1,j+1)]
        self.needlet, self.B, self.p, self.wavelet_tol = needlet, B, p, wavelet_tol

        self.spherical_basis_file = f"{self.basis_keyword}_{self.needlet}_nside{self.nside}_B{self.B}_"+ (f"p{self.p}_" if self.needlet == 'chisquare' else '') + f"tol{self.wavelet_tol}_j[{','.join([str(_i) for _i in self.j])}].h5"

        assert self.B > 1.0
        assert self.wavelet_tol >= 0.0
        assert self.needlet in ['littlewoodpaley','chisquare']
        self.S = sum([self.order_to_npix(_j) if _j >= 0 else 1 for _j in self.j])

        if self.needlet == 'chisquare':
            from SelectionFunctionUtils import chisquare
            self.weighting = chisquare(self.j, p = self.p, B = self.B, normalise=True)
        else:
            from SelectionFunctionUtils import littlewoodpaley
            self.weighting = littlewoodpaley(B = self.B)

    def _process_sigma_basis_specific(self,sigma):
        assert len(sigma) == 2
        power_spectrum = lambda l: np.exp(sigma[0]) * np.power(1.0+l,sigma[1])

        _sigma = np.zeros(self.S)
        running_index = 0
        for j in self.j:

            if j == -1:
                _sigma[running_index] = np.exp(sigma[0])
                running_index += 1
                continue

            npix_needle = self.order_to_npix(j)

            start = self.weighting.start(j)
            end = self.weighting.end(j)
            modes = np.arange(start, end + 1, dtype = 'float')
            _lambda = 4*np.pi/npix_needle # 1/np.sum(self.weighting.window_function(modes,j)* (2.0*modes+1.0)/(4*np.pi))**2 #1/npix_needle
            window = _lambda * self.weighting.window_function(modes,j)**2 * (2.0*modes+1.0)/(4*np.pi) * power_spectrum(modes)

            _sigma[running_index:running_index+npix_needle] = np.sqrt(window.sum())
            running_index += npix_needle

        # Renormalise (Marinucci 2008)
        #l = np.arange(1,self.weighting.lmax)
        #print(np.sum(power_spectrum((2*l+1)/4*np.pi * power_spectrum(l))) / np.sum(_sigma**2))
        #print(np.sum(power_spectrum((2*l+1)/4*np.pi * power_spectrum(l))) / np.sum(_sigma[1:]**2))
        #_sigma[0] = 6.9

        return _sigma

    def _generate_spherical_basis(self,gsb_file, coords=None):

        # Import dependencies
        from numba import njit
        from math import sin, cos
        import sys

        nside = self.nside
        B = self.B
        needle_sparse_tol = self.wavelet_tol

        # Function to compute needlet across sky
        @njit
        def pixel_space (Y, cos_gamma, window, start, end, legendre):
            '''Return the value of a needlet at gamma radians from the needlet centre.'''

            legendre[0] = 1.0
            legendre[1] = cos_gamma
            for cur_l in range(2, end + 1):
                legendre[cur_l] = ((cos_gamma * (2 * cur_l - 1) * legendre[cur_l - 1] - (cur_l - 1) * legendre[cur_l - 2])) / cur_l

            Y[:] = np.dot(window,legendre[start:end+1])

        # Compute locations of pixels
        if coords is None:
            npix = self.nside_to_npix(nside)
            colat, lon = np.array(hp.pix2ang(nside=nside,ipix=np.arange(npix),lonlat=False))
        else:
            colat, lon = coords
            npix = len(colat)
        cos_colat, sin_colat = np.cos(colat), np.sin(colat)
        cos_lon, sin_lon = np.cos(lon), np.sin(lon)

        # Initialise variables
        running_index = 0
        needlet_w, needlet_v, needlet_u, needlet_un, needlet_j = [], [], [], [], []
        Y = np.zeros(npix)
        legendre = np.zeros((1+self.weighting.end(max(self.j)),npix))

        for ineedlet, j in enumerate(self.j):

            print(f'Working on order {j}.')

            if j == -1:
                needlet_w.append(np.ones(npix))
                needlet_v.append(np.arange(npix))
                needlet_u.append(0)
                needlet_un.append(np.zeros(npix, dtype=np.uint64))
                needlet_j.append(np.zeros(1))
                running_index += npix
                continue

            nside_needle = self.order_to_nside(j)
            npix_needle = self.nside_to_npix(nside_needle)

            start = self.weighting.start(j)
            end = self.weighting.end(j)
            modes = np.arange(start, end + 1, dtype = 'float')
            _lambda = 4*np.pi/npix_needle # 1/np.sum(self.weighting.window_function(modes,j)* (2.0*modes+1.0)/(4*np.pi))**2 # 1/npix_needle
            window = np.sqrt(_lambda) * self.weighting.window_function(modes,j) * (2.0*modes+1.0)/(4.0*np.pi)

            for ipix_needle in tqdm.tqdm(range(npix_needle),file=sys.stdout):

                colat_needle, lon_needle = hp.pix2ang(nside=nside_needle,ipix=ipix_needle,lonlat=False)

                cos_gamma = cos(colat_needle) * cos_colat + sin(colat_needle) * sin_colat * (cos(lon_needle) * cos_lon + sin(lon_needle) * sin_lon)

                pixel_space(Y, cos_gamma = cos_gamma, window = window, start = start, end = end, legendre = legendre)

                _significant = np.where(np.abs(Y) > Y.max()*needle_sparse_tol)[0]
                needlet_w.append(Y[_significant])
                needlet_v.append(_significant)
                needlet_u.append(running_index)
                needlet_un.append(np.zeros(_significant.size, dtype=np.uint64) +j+ipix_needle )
                needlet_j.append(j*np.ones(self.order_to_npix(j)))
                running_index += _significant.size

        # Add the ending index to u
        needlet_u.append(running_index)

        # Concatenate the lists
        needlet_w = np.concatenate(needlet_w)
        needlet_v = np.concatenate(needlet_v)
        needlet_u = np.array(needlet_u)

        # Flip them round
        from scipy import sparse
        Y = sparse.csr_matrix((needlet_w,needlet_v,needlet_u)).transpose().tocsr()
        wavelet_w, wavelet_v, wavelet_u = Y.data, Y.indices, Y.indptr
        wavelet_j = np.concatenate(needlet_j).astype(int)
        wavelet_n = wavelet_w.size

        print('Expanding u')
        @njit
        def expand_u(wavelet_u, wavelet_U):
            size = wavelet_u.size-1
            for iS in range(size):
                wavelet_U[wavelet_u[iS]:wavelet_u[iS+1]] = iS
        wavelet_U = np.zeros(wavelet_v.size, dtype=np.uint64)
        expand_u(wavelet_u, wavelet_U)

        if coords is None:
            # Save file
            save_kwargs = {'compression':"lzf", 'chunks':True, 'fletcher32':False, 'shuffle':True}
            with h5py.File(gsb_file, 'w') as f:
                f.create_dataset('wavelet_w', data = wavelet_w, dtype = np.float64, **save_kwargs)
                f.create_dataset('wavelet_v', data = wavelet_v, dtype = np.uint64, scaleoffset=0, **save_kwargs)
                f.create_dataset('wavelet_u', data = wavelet_u, dtype = np.uint64, scaleoffset=0, **save_kwargs)
                f.create_dataset('wavelet_U', data = wavelet_U, dtype = np.uint64, scaleoffset=0, **save_kwargs)
                f.create_dataset('wavelet_n', data = wavelet_n)
                f.create_dataset('modes', data = wavelet_j, dtype = np.uint64, scaleoffset=0, **save_kwargs)
        else: return Y

        
    @staticmethod
    @njit(parallel=parallel)
    def wavelet_x_sparse(z, M, C, mu, sigma, wavelet, cholesky_m, cholesky_c, x):

        x *= 0.

        for iP, iS, wSP in wavelet:
            for iM, iMsub, wM in cholesky_m:
                for iC, iCsub, wC in cholesky_c:
                    x[iP,iM,iC] += wM * z[iS,iMsub,iCsub] * wC * sigma[iS] * wSP

            for iM in range(M):
                for iC in range(C):
                    x[iP,iM,iC] += mu[iS] * wSP

        return x
    
    @staticmethod
    @njit
    def wavelet_b_sparse(z, M, C, S, mu, sigma, cholesky_m, cholesky_c, b):

        b *= 0.

        # Iterate over modes which are not sparsified in Y
        for iS in range(S):
            for iM, iMsub, wM in cholesky_m:
                for iC, iCsub, wC in cholesky_c:
                    b[iS, iM, iC] += sigma[iS] * wM * z[iS,iMsub,iCsub] * wC;

            for iM in range(M):
                for iC in range(C):
                    b[iS,iM,iC] += mu[iS]

        return b


    @staticmethod
    @njit(parallel=parallel)
    def wavelet_magnitude_colour_position_sparse(z, M, C, P, k, n, mu, sigma,
                                                    wavelet_u, wavelet_v, wavelet_w,
                                                    cholesky_u_m, cholesky_v_m, cholesky_w_m,
                                                    cholesky_u_c, cholesky_v_c, cholesky_w_c,
                                                    x, lnL_grad):

        wavelet = list(zip(wavelet_u, wavelet_v, wavelet_w))
        cholesky_m = list(zip(cholesky_u_m, cholesky_v_m, cholesky_w_m))
        cholesky_c = list(zip(cholesky_u_c, cholesky_v_c, cholesky_w_c))

        x *= 0.
        for iP, iS, wSP in wavelet:
            tmpS = wSP * sigma[iS]
            for iM, iMsub, wM in cholesky_m:
                tmpM = wM * tmpS
                for iC, iCsub, wC in cholesky_c:
                    x[iP,iM,iC] += z[iS,iMsub,iCsub] * wC * tmpM

            tmpS = wSP * mu[iS]
            for iM in range(M):
                for iC in range(C):
                    x[iP,iM,iC] += tmpS
        #x = wavelet_x_sparse(z, M, C, mu, sigma, wavelet, cholesky_m, cholesky_c, x)

        d = 1 + np.exp(-np.abs(x))
        lnL = np.sum( k*x - n*(x/2 + np.abs(x)/2 + np.log(d) ) )

        lnL_grad *= 0.
        for iP, iS, wSP in wavelet:
            tmpS = wSP * sigma[iS]
            for iM, iMsub, wM in cholesky_m:
                tmpM = wM * tmpS
                for iC, iCsub, wC in cholesky_c:
                    lnL_grad[iS,iMsub,iCsub] += ( k[iP,iM,iC] - n[iP,iM,iC]*(0.5 + np.sign(x[iP,iM,iC])*(1/d[iP,iM,iC] - 0.5)) ) * wC * tmpM

        # Add on Gaussian prior
        return lnL, lnL_grad