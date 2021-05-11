#!/usr/bin/env python
#
# base.py
# The base class of all tools in SelectionFunctionToolbox.
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
import os
from .kernel import WhiteNoise

class Base:

    basis_keyword = 'base' # This must be changed in each file.

    def __init__(self, k, n, file_root, axes  = ['magnitude','colour','position'], magnitude_bins = None, colour_bins = None, magnitude_kernel = None, colour_kernel = None, sparse = False, sparse_tol = 1e-4, pivot = False, pivot_tol = 1e-4, nest = None, mu = None, sigma = None, spherical_basis_directory='./SphericalBasis',stan_model_directory='./StanModels',stan_output_directory='./StanOutput', **kwargs):

        # Utilities
        self.order_to_nside = lambda order: 2**order
        self.nside_to_npix = lambda nside: 12*nside**2
        self.order_to_npix = lambda order: self.nside_to_npix(self.order_to_nside(order))

        self.spherical_basis_directory = self._verify_directory(spherical_basis_directory)
        self.stan_model_directory = self._verify_directory(stan_model_directory)
        self.stan_output_directory = self._verify_directory(stan_output_directory)

        self.sparse = sparse
        self.sparse_tol = sparse_tol
        self.pivot = pivot
        self.pivot_tol = pivot_tol
        self.nest = nest

        # Reshape k and n to be valid
        self._reshape_k_and_n(k,n,axes)

        # Process basis-specific options
        self._process_basis_options(**kwargs)

        # Load spherical basis
        self._load_spherical_basis()

        # Process covariance kernel options
        self.magnitude_bins = magnitude_bins if magnitude_bins is not None else np.arange(self.M+1)
        self.colour_bins = colour_bins if colour_bins is not None else np.arange(self.C+1)
        self.magnitude_kernel = magnitude_kernel if magnitude_kernel is not None else WhiteNoise(self.M)
        self.colour_kernel = colour_kernel if colour_kernel is not None else WhiteNoise(self.C)

        # Compute cholesky matrices
        self.M_subspace, self.cholesky_m = self._construct_cholesky_matrix(self.magnitude_kernel,self.magnitude_bins)
        self.C_subspace, self.cholesky_c = self._construct_cholesky_matrix(self.colour_kernel,self.colour_bins)

        # Process mu and sigma
        self._process_mu(mu)
        self._process_sigma(sigma)

        # Load Stan Model
        self._load_stan_model()

        # Construct Stan Input
        self._construct_stan_input()

        # File root
        self.file_root = file_root

    def optimize(self, number_of_iterations = 1000, inits = 2):

        import time
        print('Running optimisation')
        t1 = time.time()
        _stan_optimum = self.stan_model.optimize(data = self.stan_input, iter = number_of_iterations, output_dir = self.stan_output_directory, inits = inits)
        t2 = time.time()
        print(f'Finished optimisation, it took {t2-t1:.1f} seconds')

        # Extract maxima
        _size_z = self.S*self.M_subspace*self.C_subspace
        _size_x = self.M*self.C*self.P
        self.optimum_lnp = _stan_optimum.optimized_params_np[0]
        self.optimum_z = np.transpose(_stan_optimum.optimized_params_np[1:1+_size_z].reshape((self.C_subspace,self.M_subspace,self.S)))
        self.optimum_x = np.transpose(_stan_optimum.optimized_params_np[1+_size_z:].reshape((self.P,self.C,self.M)))
        if self.nest:
            _ring_indices = hp.nest2ring(self.nside, np.arange(self.P))
            self.optimum_x = self._ring_to_nest(self.optimum_x)

        # Move convergence information somewhere useful
        import shutil
        self.optimum_convergence_file = self.file_root+'_convergence.txt'
        shutil.move(self.stan_output_directory + str(_stan_optimum).split('/')[-1], self.stan_output_directory + self.optimum_convergence_file)
        print(f'Convergence information stored in {self.stan_output_directory + self.optimum_convergence_file}')

        self.optimum_results_file = self.file_root+'_results.h5'
        self.save_h5(t2-t1)

    def save_h5(self, runtime, *keys):

        # Save optimum to h5py
        with h5py.File(self.stan_output_directory + self.optimum_results_file, 'w') as orf:
            orf.create_dataset('opt_runtime', data = runtime)
            orf.create_dataset('lnP', data = self.optimum_lnp)
            orf.create_dataset('z', data = self.optimum_z, dtype = np.float64, compression = 'lzf', chunks = True)
            orf.create_dataset('x', data = self.optimum_x, dtype = np.float64, compression = 'lzf', chunks = True)
            orf.create_dataset('magnitude_bins', data = self.magnitude_bins, dtype = np.float64)
            orf.create_dataset('colour_bins', data = self.colour_bins, dtype = np.float64)
            for key in keys:
                try: orf.create_dataset(key, data = getattr(self, key), dtype = np.float64)
                except AttributeError: print(f'No attribute: {key}')
        print(f'Optimum values stored in {self.stan_output_directory + self.optimum_results_file}')

    def sample(self, number_of_iterations=1000, number_of_warmups=100, threads=5, chains=4, inits=2):

        #_stan_samples = self.stan_model.sample(data=self.stan_input, chains=chains, parallel_chains=chains, show_progress=True,
        #                                       iter_warmup=number_of_warmups, iter_sampling=number_of_iterations, output_dir=self.stan_output_directory)

        _stan_samples = self.stan_model.sample(data=self.stan_input, chains=chains, parallel_chains=chains, threads_per_chain=threads, show_progress=True, #refresh=10,
                                              iter_warmup=number_of_warmups, iter_sampling=number_of_iterations, output_dir=self.stan_output_directory)

    def print_convergence(self, number_of_lines = 2):
        for line in self._tail(self.stan_output_directory + self.optimum_convergence_file,number_of_lines):
            print(line)

    def _verify_directory(self,_directory):

        # Check it exists, if not then create
        if not os.path.exists(_directory):
            os.makedirs(_directory)

        # Ensure it ends with '/'
        if _directory[-1] != '/':
            _directory = _directory + '/'

        return _directory

    def _process_basis_options(self,options):
        pass

    def _reshape_k_and_n(self,k,n,axes):

        possible_axes = ["magnitude","colour","position"]
        axes_size = len(axes)

        assert 'position' in axes # We do not yet accept magnitude-colour only selection functions
        assert set(axes).issubset(set(possible_axes)) # We can only accept magnitude, colour and position
        assert axes_size == len(set(axes)) # Each axis must be unique
        assert k.shape == n.shape # k and n must have same shape

        new_indices = [possible_axes.index(axis) for axis in axes]
        self.k = np.moveaxis(k.copy().reshape(k.shape+(1,)*(3-axes_size)),range(axes_size),new_indices)
        self.n = np.moveaxis(n.copy().reshape(n.shape+(1,)*(3-axes_size)),range(axes_size),new_indices)

        self.M, self.C, self.P = self.k.shape
        if self.nest != None:
            assert hp.isnpixok(self.P) # number of pixels must be valid if we are using a healpix position basis

        if self.nest == True:
            self.k = self._nest_to_ring(self.k)
            self.n = self._nest_to_ring(self.n)

    def _load_spherical_basis(self):
        """ Loads in the spherical basis file. If they don't exist, then generate them. The generator must be implemented in each child class. """

        _spherical_basis_files = self.spherical_basis_file

        if not os.path.isfile(self.spherical_basis_directory + self.spherical_basis_file):
            print('Spherical basis file does not exist, generating... (this may take some time!)')
            self._generate_spherical_basis(self.spherical_basis_directory + self.spherical_basis_file)

        # Load spherical basis
        with h5py.File(self.spherical_basis_directory + self.spherical_basis_file, 'r') as sbf:
            self.basis = {k:v[()] for k,v in sbf.items()}

        print('Spherical basis file loaded')

    def _construct_cholesky_matrix(self, kernel, bins):

        # Compute cholesky
        _x = 0.5*(bins[1:]+bins[:-1])
        _d = np.abs(_x[:,np.newaxis]-_x[np.newaxis,:])
        _N_subspace, _cholesky = kernel.cholesky(_d)

        return _N_subspace, _cholesky

    def _process_mu(self,mu):

        # Process mu
        if mu == None:
            self.mu = 0.0
        else:
            self.mu = mu

    def _process_sigma(self,sigma):

        # Process sigma
        if sigma == None:
            self.sigma = np.ones(self.S)
        elif isinstance(sigma, np.ndarray):
            assert sigma.shape == (self.S,)
            self.sigma = sigma
        elif callable(sigma):
            self.sigma = sigma(self.basis['modes'])
        elif type(sigma) in [list,tuple]:
            self.sigma = self._process_sigma_basis_specific(sigma)
        else:
            self.sigma = sigma*np.ones(self.S)

    def _process_mu_basis_specific(self,sigma):
        pass

    def _process_sigma_basis_specific(self,sigma):
        pass


    def _load_stan_model(self):

        _model_file = self.stan_model_directory+f"{self.basis_keyword}_magnitude_colour_position{'_sparse' if self.sparse else ''}.stan"

        if not os.path.isfile(_model_file):
            with open(_model_file, 'w') as f:
                f.write(self._yield_stan_model())

        from cmdstanpy import CmdStanModel
        self.stan_model = CmdStanModel(stan_file = _model_file, cpp_options={"STAN_THREADS": True})

    def _yield_stan_model(self):
        pass

    def _construct_stan_input(self):

        self.stan_input = {'k':self.k,
                           'n':self.n,
                           'P':self.P,
                           'M':self.M,
                           'M_subspace':self.M_subspace,
                           'C':self.C,
                           'C_subspace':self.C_subspace,
                           'S':self.S,
                           'mu':self.mu,
                           'sigma':self.sigma}

        # Add the basis specific keys
        self.stan_input.update(self.basis)

        if self.sparse:
            self.stan_input['cholesky_n_m'], self.stan_input['cholesky_w_m'], self.stan_input['cholesky_v_m'], self.stan_input['cholesky_u_m'] = self._sparsify(self.cholesky_m)
            self.stan_input['cholesky_n_c'], self.stan_input['cholesky_w_c'], self.stan_input['cholesky_v_c'], self.stan_input['cholesky_u_c'] = self._sparsify(self.cholesky_c)
        else:
            self.stan_input['cholesky_m'] = self.cholesky_m
            self.stan_input['cholesky_c'] = self.cholesky_c

        integer_types = (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)
        for k,v in self.stan_input.items():

            # Convert everything to numpy array
            if isinstance(v, (list, tuple)):
                v = np.array(v)

            # Correct for Stan being one-indexed - only apply to one-dimensional collections of integers
            if isinstance(v, np.ndarray) and v.dtype in integer_types:
                if k not in ['n','k','modes']:
                    print('Incrementing',k)
                    v = v + 1

            # Deal with serialisation issue with integers
            if isinstance(v, integer_types):
                v = v.item()

            self.stan_input[k] = v



    def _sparsify(self,_matrix):

        # Set any elements in each row that are smaller than self.sparse_tol * max(row) to zero
        _height,_width = _matrix.shape
        _sparse_matrix = _matrix.copy()

        for n in range(_height):
            _row = np.abs(_matrix[n])
            _sparse_matrix[n,_row/max(_row) < self.sparse_tol] = 0

        # Compute the CSR decomposition of the sparse matrix
        from scipy.sparse import csr_matrix
        _csr_matrix = csr_matrix(_sparse_matrix)
        _csr_n = _csr_matrix.data.size
        print(f"{100*(1.0-_csr_n/(_height*_width)):.2f}% sparsity")

        return _csr_n, _csr_matrix.data, _csr_matrix.indices, _csr_matrix.indptr


    def _tail(self,filename, lines=1, _buffer=4098):
        """Tail a file and get X lines from the end"""
        # place holder for the lines found
        lines_found = []

        # block counter will be multiplied by buffer
        # to get the block size from the end
        block_counter = -1

        with open(filename,'r') as f:
            # loop until we find X lines
            while len(lines_found) < lines:
                try:
                    f.seek(block_counter * _buffer, os.SEEK_END)
                except IOError:  # either file is too small, or too many lines requested
                    f.seek(0)
                    lines_found = f.readlines()
                    break

                lines_found = f.readlines()

                block_counter -= 1

        return lines_found[-lines:]

    def _nest_to_ring(self,A):
        """ Reorders an array of shape (M,C,P) to ring ordering. """
        _npix = A.shape[2]
        _nside = hp.npix2nside(_npix)
        _reordering = hp.ring2nest(_nside, np.arange(_npix))
        return A[:,:,_reordering]

    def _ring_to_nest(self,A):
        """ Reorders an array of shape (M,C,P) to ring ordering. """
        _npix = A.shape[2]
        _nside = hp.npix2nside(_npix)
        _reordering = hp.nest2ring(_nside, np.arange(_npix))
        return A[:,:,_reordering]

    def _generate_spherical_basis(self,gsb_file):
        pass
