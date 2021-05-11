#!/usr/bin/env python
#
# Models a selection function that is independent between positions.
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

class Wrench(Base):

    basis_keyword = 'basic'

    def _process_basis_options(self):
        self.S = self.P
        self.spherical_basis_file = f'{self.basis_keyword}.h5'

    def _generate_spherical_basis(self,gsb_file):
        with h5py.File(gsb_file, 'w') as f:
            pass

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
            x[m,c,s] = mu + sigma[s] * (cholesky_w_m[cholesky_u_m[m]:cholesky_u_m[m+1]-1] * z[s,cholesky_v_m[cholesky_u_m[m]:cholesky_u_m[m+1]-1], cholesky_v_c[cholesky_u_c[c]:cholesky_u_c[c+1]-1]] * cholesky_w_c[cholesky_u_c[c]:cholesky_u_c[c+1]-1]);
            '''
        else:
            cholesky_parameters = '''
            row_vector[M_subspace] cholesky_m[M]; // Cholesky factor in magnitude space
            vector[C_subspace] cholesky_c[C];     // Cholesky factor in colour space
            '''
            cholesky_loop = '''
            x[m,c,s] = mu + sigma[s] * cholesky_m[m] * z[s] * cholesky_c[c];
            '''

        stan_model = f'''
        data {{
            int<lower=0> P;                       // number of pixels
            int<lower=0> M;                       // number of bins in magnitude space
            int<lower=0> M_subspace;              // number of inducing points in magnitude space
            int<lower=0> C;                       // number of bins in colour space
            int<lower=0> C_subspace;              // number of inducing points in colour space
            int<lower=0> S;                       // number of modes
            real mu;                              // mean across sky
            vector[S] sigma;                      // sigma of each mode
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
                    // Compute x
                    for (s in 1:S){{
                        {cholesky_loop}
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