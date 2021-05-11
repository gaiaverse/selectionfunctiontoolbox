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
import copy

class kernelbase:

    def __init__(self, variance = 1.0, pivot = False, pivot_tol = 0.0, eps_tol = 1e-15, **kwargs):
        self.variance = variance
        self.pivot = pivot
        self.pivot_tol = pivot_tol
        self.eps_tol = eps_tol
        self._process_kernel_arguments(**kwargs)

    def distance(self, x1, x2):
        return np.abs(x1[:,np.newaxis]-x2[np.newaxis,:])

    def covariance(self, x1, x2):
        pass

    def cholesky(self, x):

        _covariance = self.covariance(x, x)
        _n = x.shape[0]

        if self.pivot:
            _cholesky = self._pivoted_cholesky(_covariance, M=_n, err_tol=self.pivot_tol)
        else:
            _cholesky = np.linalg.cholesky(_covariance+self.eps_tol*np.diag(np.ones(_n)))

        _n_subspace = _cholesky.shape[1]

        return _n_subspace, _cholesky

    def __add__(self, other):

        if isinstance(other,(float,int)):
            _new_kernel = AdditiveKernel(kernel1=self,kernel2=Flat(variance=other))
        elif isinstance(other,kernelbase):
            _new_kernel = AdditiveKernel(kernel1=self,kernel2=other)
        return _new_kernel

    __radd__ = __add__

    def __mul__(self, other):

        if isinstance(other,(float,int)):
            _new_kernel = copy.deepcopy(self)
            _new_kernel.variance = self.variance*other
        elif isinstance(other,kernelbase):
            _new_kernel = MultiplicativeKernel(kernel1=self,kernel2=other)
        return _new_kernel

    __rmul__ = __mul__


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

class AdditiveKernel(kernelbase):

    def covariance(self, x1, x2):
        return self.variance*(self.kernel1.covariance(x1, x2)+self.kernel2.covariance(x1, x2))

    def _process_kernel_arguments(self, kernel1, kernel2):
        self.kernel1 = copy.deepcopy(kernel1)
        self.kernel2 = copy.deepcopy(kernel2)
        self.pivot = kernel1.pivot or kernel2.pivot
        self.pivot_tol = max(kernel1.pivot_tol,kernel2.pivot_tol)

class MultiplicativeKernel(kernelbase):

    def covariance(self, x1, x2):
        return self.variance*self.kernel1.covariance(x1, x2)*self.kernel2.covariance(x1, x2)

    def _process_kernel_arguments(self, kernel1, kernel2):
        self.kernel1 = copy.deepcopy(kernel1)
        self.kernel2 = copy.deepcopy(kernel2)
        self.pivot = kernel1.pivot or kernel2.pivot
        self.pivot_tol = max(kernel1.pivot_tol,kernel2.pivot_tol)

class SquaredExponential(kernelbase):

    def covariance(self, x1, x2):
        distance = self.distance(x1, x2)
        return self.variance*np.exp(-0.5*np.square(distance/self.lengthscale))

    def _process_kernel_arguments(self, lengthscale = 1.0):
        self.lengthscale = lengthscale

class Matern12(kernelbase):

    def covariance(self, x1, x2):
        distance = self.distance(x1, x2)
        return self.variance*np.exp(-distance/self.lengthscale)

    def _process_kernel_arguments(self, lengthscale = 1.0):
        self.lengthscale = lengthscale

class Matern32(kernelbase):

    def covariance(self, x1, x2):
        distance = self.distance(x1, x2)
        u = np.sqrt(3.0)*distance/self.lengthscale
        return self.variance*np.exp(-u)*(1.0+u)

    def _process_kernel_arguments(self, lengthscale = 1.0):
        self.lengthscale = lengthscale

class RationalQuadratic(kernelbase):

    def covariance(self, x1, x2):
        distance = self.distance(x1, x2)
        return self.variance*np.power(1.0+0.5*np.square(distance/self.lengthscale)/self.mixturescale,-self.mixturescale)

    def _process_kernel_arguments(self, lengthscale = 1.0, mixturescale = 1.0):
        self.lengthscale = lengthscale
        self.mixturescale = mixturescale

class Periodic(kernelbase):

    def covariance(self, x1, x2):
        distance = self.distance(x1, x2)
        return self.variance*np.exp(-2.0*np.square(np.sin(np.pi*distance/self.period)/self.lengthscale))

    def _process_kernel_arguments(self, lengthscale = 1.0, period = 1.0):
        self.lengthscale = lengthscale
        self.period = period

class Linear(kernelbase):

    def covariance(self, x1, x2):
        return self.variance*(x1[:,np.newaxis]-self.intercept)*(x2[np.newaxis,:]-self.intercept)

    def _process_kernel_arguments(self, intercept = 0.0):
        self.intercept = intercept

class WhiteNoise(kernelbase):

    def covariance(self, x1, x2):
        distance = self.distance(x1, x2)
        covariance = np.zeros_like(distance)
        covariance[np.isclose(distance,0.0)] = self.variance
        return covariance

class Flat(kernelbase):

    def covariance(self, x1, x2):
        return self.variance*np.ones((x1.size,x2.size))