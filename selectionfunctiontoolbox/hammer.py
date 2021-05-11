#!/usr/bin/env python
#
# hammer.py
# Uses spherical harmonics to estimate a selection function across the sky.
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
import os

class Hammer(Base):

    basis_keyword = 'harmonic'

    def _process_basis_options(self,lmax = 0):
        self.lmax = lmax
        self.S = (self.lmax + 1) ** 2
        self.nside = hp.npix2nside(self.P)
        self.spherical_basis_file = f'{self.basis_keyword}_nside{self.nside}_lmax{self.lmax}.h5'

    def _process_sigma_basis_specific(self,sigma):
        assert len(sigma) == 2
        return np.sqrt(np.exp(sigma[0])*np.power(1.0+self.basis['modes'],sigma[1]))

    def _generate_spherical_basis(self,gsb_file):

        nside = self.nside
        lmax = self.lmax
        Npix = self.P

        # Form the l's and m's
        Nmodes = int((lmax+1)**2)
        Nmodes_hp = int((lmax+1)*(lmax+2)/2)
        l_hp,m_hp = hp.sphtfunc.Alm.getlm(lmax=lmax)
        assert Nmodes_hp == l_hp.size

        l, m = np.zeros(Nmodes,dtype=int), np.zeros(Nmodes,dtype=int)
        l[:Nmodes_hp],m[:Nmodes_hp] = l_hp,m_hp
        l[Nmodes_hp:],m[Nmodes_hp:] = l_hp[lmax+1:],-m_hp[lmax+1:]

        # Ring idxs of pixels with phi=0
        theta, phi = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
        theta_ring, unique_idx, jpix = np.unique(theta, return_index=True, return_inverse=True)

        # Generate lambda
        _lambda = np.zeros((Nmodes, 4*nside-1))
        if False: # From scipy
            # For |m|>0 this comes out a factor of 2 smaller than the healpy version
            # For m<0 there's also a factor of (-1)^m difference
            for i,(_l,_m) in enumerate(zip(tqdm.tqdm(l),m)):
                _lambda[i] = (-1)**np.abs(_m) * np.real( scipy.special.sph_harm(np.abs(_m), _l, theta_ring*0., theta_ring) )
        else: # From healpy
            alm_hp = np.zeros(Nmodes_hp)
            for i,(_l,_m) in enumerate(zip(tqdm.tqdm(l),m)):
                i_hp = hp.sphtfunc.Alm.getidx(lmax, _l, np.abs(_m))
                alm_hp = np.zeros(Nmodes_hp)*(0.+0.j)
                # Get real component
                alm_hp[i_hp] = 1.+0.j
                map_hp = (1.+0.j)*hp.sphtfunc.alm2map(alm_hp,nside=nside, verbose=False)
                # Add imaginary component
                alm_hp[i_hp] = 0.+1.j
                map_hp += (0.-1.j)*hp.sphtfunc.alm2map(alm_hp,nside=nside, verbose=False)
                alm_hp[i_hp] = 0.+0.j
                map_hp /= np.exp(1.j*np.abs(_m)*phi)
                # Select unique latitude indices
                _lambda[i] = (-1)**np.abs(_m) * np.real(map_hp)[unique_idx]

                # Divide by 2
                if _m != 0:
                    _lambda[i] /= 2.0

        # Generate Exponential
        azimuth = np.ones((2*lmax+1,Npix))
        for _m in range(-lmax, lmax+1):
            if _m<0:   azimuth[_m+lmax] = np.sqrt(2) * np.sin(-_m*phi)
            elif _m>0: azimuth[_m+lmax] = np.sqrt(2) * np.cos(_m*phi)
            else: pass

        # Generate indices mapping m to alm
        lower, upper = np.zeros(2*lmax+1),np.zeros(2*lmax+1)
        for i, _m in enumerate(range(-lmax,lmax+1)):
            match = np.where(m==_m)[0]
            lower[i] = match[0]
            upper[i] = match[-1]

        save_kwargs = {'compression':"lzf", 'chunks':True, 'fletcher32':False, 'shuffle':True}
        with h5py.File(gsb_file, 'w') as f:
            # Create datasets
            f.create_dataset('lambda', data = _lambda.T, shape = (4*nside-1, Nmodes,), dtype = np.float64, **save_kwargs)
            f.create_dataset('azimuth',data = azimuth, shape = (2*lmax+1, Npix, ),   dtype = np.float64, **save_kwargs)
            f.create_dataset('modes',      data = l,       shape = (Nmodes,), dtype = np.uint32, scaleoffset=0, **save_kwargs)
            f.create_dataset('submodes',      data = m,       shape = (Nmodes,), dtype = np.int32, scaleoffset=0, **save_kwargs)
            f.create_dataset('pixel_to_ring',   data = jpix.astype(int),    shape = (Npix,),   dtype = np.uint32, scaleoffset=0, **save_kwargs)
            f.create_dataset('lower',   data = lower.astype(int),    shape = (2*lmax+1,),   dtype = np.uint32, scaleoffset=0, **save_kwargs)
            f.create_dataset('upper',   data = upper.astype(int),    shape = (2*lmax+1,),   dtype = np.uint32, scaleoffset=0, **save_kwargs)
            f.create_dataset('L', data = 2 * self.lmax + 1)
            f.create_dataset('R', data = 4 * self.nside - 1)

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
            a[s] = sigma[s] * (cholesky_w_m[cholesky_u_m[m]:cholesky_u_m[m+1]-1] * z[s,cholesky_v_m[cholesky_u_m[m]:cholesky_u_m[m+1]-1], cholesky_v_c[cholesky_u_c[c]:cholesky_u_c[c+1]-1]] * cholesky_w_c[cholesky_u_c[c]:cholesky_u_c[c+1]-1]);
            '''
        else:
            cholesky_parameters = '''
            row_vector[M_subspace] cholesky_m[M]; // Cholesky factor in magnitude space
            vector[C_subspace] cholesky_c[C];     // Cholesky factor in colour space
            '''
            cholesky_loop = '''
            a[s] = sigma[s] * cholesky_m[m] * z[s] * cholesky_c[c];
            '''

        stan_model = f'''
        data {{
            int<lower=0> P;                       // number of pixels
            int<lower=0> M;                       // number of bins in magnitude space
            int<lower=0> M_subspace;              // number of inducing points in magnitude space
            int<lower=0> C;                       // number of bins in colour space
            int<lower=0> C_subspace;              // number of inducing points in colour space
            int<lower=0> L;                       // 2 * max l of hamonics + 1
            int<lower=0> S;                       // number of harmonics
            int<lower=0> R;                       // number of HEALPix isolatitude rings
            matrix[R,S] lambda;                   // spherical harmonics decomposed
            matrix[L,P] azimuth;                  // spherical harmonics decomposed
            int pixel_to_ring[P];                 // map P->R
            int lower[L];                         // map S->L
            int upper[L];                         // map S->L
            real mu;                              // mean across sky
            vector[S] sigma;                      // sigma of each harmonic
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
                    
                    // Local variables
                    vector[S] a;
                    matrix[R,L] F;

                    // Compute a 
                    for (s in 1:S){{
                        {cholesky_loop}
                    }}

                    // Compute F
                    for (l in 1:L) {{
                        F[:,l] = lambda[:,lower[l]:upper[l]] * a[lower[l]:upper[l]];
                    }}

                    // Compute x
                    for (p in 1:P){{
                        x[m,c,p] = mu + dot_product(F[pixel_to_ring[p]],azimuth[:,p]);
                    }}

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
