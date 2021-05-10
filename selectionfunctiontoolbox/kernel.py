#!/usr/bin/env python
#
# kernel.py
# Covariance kernels for the magnitude and colour dimensions.
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

class kernelbase:

    def __init__(self, pivot = False, pivot_tol = None, eps_tol = 1e-15, **kwargs):
        self.pivot = pivot
        self.pivot_tol = pivot_tol if pivot_tol != None else 0.0
        self._process_kernel_arguments(**kwargs)

    def covariance(self, distance):
        pass

    def cholesky(self, distance):

        _covariance = self.covariance(distance)
        _n = distance.shape[0]

        if self.pivot:
            _cholesky = self._pivoted_cholesky(_covariance, M=_n, err_tol=self.pivot_tol)
        else:
            _cholesky = np.linalg.cholesky(_covariance+self.eps_tol*np.diag(np.ones(_n)))

        _n_subspace = _cholesky.shape[1]

        return _n_subspace, _cholesky

    def _process_kernel_arguments(self):
        pass

    def _pivoted_cholesky(self, A, M, err_tol = 1e-6):
        """
        https://dl.acm.org/doi/10.1016/j.apnum.2011.10.001 implemented by https://github.com/NathanWycoff/PivotedCholesky
        A simple python function which computes the Pivoted Cholesky decomposition/approximation of positive semi-definite operator. Only diagonal elements and select rows of that operator's matrix represenation are required.
        get_diag - A function which takes no arguments and returns the diagonal of the matrix when called.
        get_row - A function which takes 1 integer argument and returns the desired row (zero indexed).
        M - The maximum rank of the approximate decomposition; an integer.
        err_tol - The maximum error tolerance, that is difference between the approximate decomposition and true matrix, allowed. Note that this is in the Trace norm, not the spectral or frobenius norm.
        Returns: R, an upper triangular matrix of column dimension equal to the target matrix. It's row dimension will be at most M, but may be less if the termination condition was acceptably low error rather than max iters reached.
        """

        get_diag = lambda: np.diag(A).copy()
        get_row = lambda i: A[i,:]

        d = np.copy(get_diag())
        N = len(d)

        pi = list(range(N))

        R = np.zeros([M,N])

        err = np.sum(np.abs(d))

        m = 0
        while (m < M) and (err > err_tol):

            i = m + np.argmax([d[pi[j]] for j in range(m,N)])

            tmp = pi[m]
            pi[m] = pi[i]
            pi[i] = tmp

            R[m,pi[m]] = np.sqrt(d[pi[m]])
            Apim = get_row(pi[m])
            for i in range(m+1, N):
                if m > 0:
                    ip = np.inner(R[:m,pi[m]], R[:m,pi[i]])
                else:
                    ip = 0
                R[m,pi[i]] = (Apim[pi[i]] - ip) / R[m,pi[m]]
                d[pi[i]] -= pow(R[m,pi[i]],2)

            err = np.sum([d[pi[i]] for i in range(m+1,N)])
            m += 1

        R = R[:m,:]

        return(R.T)

class SquaredExponentialKernel(kernelbase):

    def covariance(self, distance):
        return self.variance*np.exp(-0.5*np.square(distance/self.lengthscale))

    def _process_kernel_arguments(self, lengthscale = None, variance = None):
        self.lengthscale = lengthscale if lengthscale != None else 1.0
        self.variance = variance if variance != None else 1.0

class Matern12(kernelbase):

    def covariance(self, distance):
        return self.variance*np.exp(-distance/self.lengthscale)

    def _process_kernel_arguments(self, lengthscale = None, variance = None):
        self.lengthscale = lengthscale if lengthscale != None else 1.0
        self.variance = variance if variance != None else 1.0

class Matern32(kernelbase):

    def covariance(self, distance):
        u = np.sqrt(3.0)*distance/self.lengthscale
        return self.variance*np.exp(-u)*(1.0+u)

    def _process_kernel_arguments(self, lengthscale = None, variance = None):
        self.lengthscale = lengthscale if lengthscale != None else 1.0
        self.variance = variance if variance != None else 1.0

class IdentityKernel(kernelbase):

    def covariance(self, distance):
        covariance = np.zeros_like(distance)
        covariance[np.isclose(distance,0.0)] = 1.0
        return covariance