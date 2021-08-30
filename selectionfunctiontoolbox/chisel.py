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
import copy
from .base import Base

from PythonModels.wavelet_magnitude_colour_position import wavelet_magnitude_colour_position_sparse, wavelet_x_sparse, wavelet_b_sparse

class Chisel(Base):

    basis_keyword = 'wavelet'

    def _process_basis_options(self, needlet = 'littlewoodpaley', j=[0], B = 2.0, p = 1.0, wavelet_tol = 1e-10):

        if type(j) in [list,tuple,np.ndarray]:
            self.j = sorted([int(_j) for _j in j])
        else:
            self.j = [_j for _j in range(-1,j+1)]
        self.needlet, self.B, self.p, self.wavelet_tol = needlet, B, p, wavelet_tol

        assert self.B > 1.0
        assert self.wavelet_tol >= 0.0
        assert self.needlet in ['littlewoodpaley','chisquare']
        self.S = sum([self.order_to_npix(_j) if _j >= 0 else 1 for _j in self.j])
        self.nside = hp.npix2nside(self.P)

        self.spherical_basis_file = f"{self.basis_keyword}_{self.needlet}_nside{self.nside}_B{self.B}_"+ (f"p{self.p}_" if self.needlet == 'chisquare' else '') + f"tol{self.wavelet_tol}_j[{','.join([str(_i) for _i in self.j])}].h5"

        if self.needlet == 'chisquare':
            from .utils import chisquare
            self.weighting = chisquare(self.j, p = self.p, B = self.B)
        else:
            from .utils import littlewoodpaley
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
        else:
            return Y

    def _yield_stan_model(self):
        if self.sparse:
            cholesky_parameters = '''
            int cholesky_n_m;                     // sparse cholesky in magnitude - number of nonzero elements
            row_vector[cholesky_n_m] cholesky_w_m;// sparse cholesky in magnitude - nonzero elements
            int cholesky_v_m[cholesky_n_m];       // sparse cholesky in magnitude - columns of nonzero elements
            int cholesky_u_m[M+1];                // sparse cholesky in magnitude - where in w each row starts
            int cholesky_n_c;                     // sparse cholesky in colour - number of nonzero elements
            vector[cholesky_n_c] cholesky_w_c;    // sparse cholesky in colour - nonzero elements
            int cholesky_v_c[cholesky_n_c];       // sparse cholesky in colour - columns of nonzero elements
            int cholesky_u_c[C+1];                // sparse cholesky in colour - where in w each row starts
            '''
            cholesky_loop = '''
            b[s] = sigma[s] * (cholesky_w_m[cholesky_u_m[m]:cholesky_u_m[m+1]-1] * z[s,cholesky_v_m[cholesky_u_m[m]:cholesky_u_m[m+1]-1], cholesky_v_c[cholesky_u_c[c]:cholesky_u_c[c+1]-1]] * cholesky_w_c[cholesky_u_c[c]:cholesky_u_c[c+1]-1]);
            '''
        else:
            cholesky_parameters = '''
            row_vector[M_subspace] cholesky_m[M]; // Cholesky factor in magnitude space
            vector[C_subspace] cholesky_c[C];     // Cholesky factor in colour space
            '''
            cholesky_loop = '''
            b[s] = sigma[s] * cholesky_m[m] * z[s] * cholesky_c[c];
            '''

        stan_model = f'''
        data {{
            int<lower=0> P;                       // number of pixels
            int<lower=0> M;                       // number of bins in magnitude space
            int<lower=0> M_subspace;              // number of inducing points in magnitude space
            int<lower=0> C;                       // number of bins in colour space
            int<lower=0> C_subspace;              // number of inducing points in colour space
            int<lower=0> S;                       // number of wavelets
            int wavelet_n;                        // sparse wavelets - number of nonzero elements
            vector[wavelet_n] wavelet_w;          // sparse wavelets - nonzero elements
            int wavelet_v[wavelet_n];             // sparse wavelets - columns of nonzero elements
            int wavelet_u[P+1];                   // sparse wavelets - where in w each row starts
            real mu;                              // mean across sky
            vector[S] sigma;                      // sigma of each wavelet
            int k[M,C,P];                         // number of heads
            int n[M,C,P];                         // number of flips
            {cholesky_parameters}
        }}
        parameters {{
            matrix[M_subspace,C_subspace] z[S];
        }}
        transformed parameters {{

            vector[P] x[M,C]; // Probability in logit-space

            // Loop over magnitude and colour
            for (m in 1:M){{
                for (c in 1:C){{

                    // Local variable
                    vector[S] b;

                    // Compute b
                    for (s in 1:S){{
                        {cholesky_loop}
                    }}

                    // Compute x
                    x[m,c] = mu + csr_matrix_times_vector(P, S, wavelet_w, wavelet_v, wavelet_u, b);

                }}
            }}

        }}
        model {{

            // Prior
            for (s in 1:S){{
                to_vector(z[s]) ~ std_normal();
            }}

            // Likelihood
            for (m in 1:M){{
                for (c in 1:C){{
                    k[m,c] ~ binomial_logit(n[m,c], x[m,c]);
                }}
            }}

        }}
        '''

        return stan_model

    def _cholesky_args(self, sparse=False):

        if sparse:
            cholesky_u_m = np.zeros(len(self.model_input['cholesky_v_m']), dtype=int)
            for iS, iY in enumerate(self.model_input['cholesky_u_m'][1:]-1):
                cholesky_u_m[iY:] += 1
            cholesky_u_c = np.zeros(len(self.model_input['cholesky_v_c']), dtype=int)
            for iS, iY in enumerate(self.model_input['cholesky_u_c'][1:]-1):
                cholesky_u_c[iY:] += 1

            cholesky_args = [cholesky_u_m,
                             self.model_input['cholesky_v_m']-1,
                             self.model_input['cholesky_w_m'],
                             cholesky_u_c,
                             self.model_input['cholesky_v_c']-1,
                             self.model_input['cholesky_w_c']]

        elif not sparse:
            cholesky_args = [self.cholesky_m, self.cholesky_c]

        return cholesky_args

    def _construct_scipy_input(self, sparse=False):

        if not sparse: raise ValueError(f'Sparse={sparse}. Only sparse model implemented for Chisel with scipy.')

        cholesky_args = self._cholesky_args(sparse=sparse)

        lnL_grad = np.zeros((self.S, self.M, self.C))
        x = np.zeros((self.M, self.C))

        self.scipy_args = [np.moveaxis(self.k, -1,0).astype(np.int64).copy(),
                           np.moveaxis(self.n, -1,0).astype(np.int64).copy()] \
                          + [np.zeros(self.S)+copy.copy(self.model_input[arg]) for arg in ['mu', 'sigma']]\
                          + [self.model_input['wavelet_U'].copy()-1,
                             self.model_input['wavelet_v'].copy()-1,
                             self.model_input['wavelet_w'].copy()]\
                          + cholesky_args

        if sparse: self.likelihood_function = wavelet_magnitude_colour_position_sparse

    def _parallelize_scipy_model(self, nsets=1):

        P_set = np.zeros(nsets, dtype=int) + self.P//nsets
        P_set[:self.P - np.sum(P_set)] += 1
        print('P sets: ', P_set, self.P)

        wavelet_u = self.model_input['wavelet_U'].copy()-1
        wavelet_v = self.model_input['wavelet_v'].copy()-1
        wavelet_w = self.model_input['wavelet_w'].copy()

        self.scipy_args_set = []
        iP = 0
        for iset in range(nsets):

            lnL_grad = np.zeros((self.S, self.M, self.C))
            x = np.zeros((self.M, self.C))

            scipy_args_set  = [self.scipy_args[i][iP:iP+P_set[iset]].copy() for i in [0,1]] \
                              + [self.scipy_args[i] for i in [2,3]] \
                              + [self.scipy_args[i][int(self.model_input['wavelet_u'][iP]-1):int(self.model_input['wavelet_u'][iP+P_set[iset]]-1)].copy() \
                                                                        for i in [4,5,6]] \
                              + [self.scipy_args[i] for i in range(7,len(self.scipy_args))]
            scipy_args_set[4]-=iP

            self.scipy_args_set.append(scipy_args_set)
            iP += P_set[iset]

        self.P_set = P_set

    def _get_b(self, z):

        from numba.typed import List

        if not hasattr(self, 'scipy_args'): self._load_scipy_model(sparse=self.sparse)

        print('Setting cholesky arguments')
        cholesky_m = List(zip(*self.scipy_args[7:10]))
        cholesky_c = List(zip(*self.scipy_args[10:13]))

        b = np.zeros((self.S, self.M, self.C))
        b = wavelet_b_sparse(z.reshape((self.S, self.M_subspace, self.C_subspace)), self.M, self.C, self.S, *self.scipy_args[2:4], cholesky_m, cholesky_c, b)

        return b

    def _get_x(self, z):

        from numba.typed import List

        if not hasattr(self, 'scipy_args'): self._load_scipy_model(sparse=self.sparse)

        print('Setting cholesky arguments')
        wavelet = List(zip(*self.scipy_args[4:7]))
        cholesky_m = List(zip(*self.scipy_args[7:10]))
        cholesky_c = List(zip(*self.scipy_args[10:13]))

        x = np.zeros((self.P, self.M, self.C))
        x = wavelet_x_sparse(z.reshape((self.S, self.M_subspace, self.C_subspace)), self.M, self.C, *self.scipy_args[2:4], wavelet, cholesky_m, cholesky_c, x)

        return x
