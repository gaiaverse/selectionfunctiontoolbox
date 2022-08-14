import sys, os
import numpy as np
import healpy as hp
import tqdm, h5py, time
# import ray
import multiprocessing, numba
import scipy.optimize
import scipy.sparse

from SelectionFunctionBase import Base
from SelectionFunctionChisel import Chisel

from wavelet_magnitude_colour_position import wavelet_magnitude_colour_position, wavelet_magnitude_colour_position_sparse, wavelet_x_sparse, wavelet_b_sparse

global lnlike_iter
global gnorm_iter
global tinit
global evaluators

def fcall(X):
    global tinit
    global lnlike_iter
    global gnorm_iter
    print(f't={int(time.time()-tinit):05d}, lnL={lnlike_iter:.0f}, gnorm={gnorm_iter:.0f}, mean={np.mean(X):.1f}, std={np.std(X):.3f}')

def print_log(message, logfile="logs/default_log.txt"):

    if os.path.exists(logfile): mode='a'
    else: mode='w'

    with open(logfile, mode) as f:
        f.write(message+'\n')
    print(message)


#@ray.remote
class evaluate():

    def __init__(self, P, S, M, C, M_subspace, C_subspace, wavelet_args, j=[-1],
                       logfile='PyOutput/log.txt',
                       savefile='PyOutput/progress.h'):

        self.P=P
        self.S=S
        self.M=M
        self.C=C
        self.M_subspace=M_subspace
        self.C_subspace=C_subspace
        self.wavelet_args=wavelet_args

        self.j=j

        self.logfile=logfile
        self.savefile=savefile

        self.x = np.zeros((self.P, self.M, self.C))
        self.lnL_grad = np.zeros((self.S, self.M_subspace, self.C_subspace))

        self.tinit = time.time()
        self.fshift = 1.
        self.lnlike_iter = 0.
        self.gnorm_iter = 0.
        self.nfev = 0
        self.nfev_print = -100
        self.lnlike_history = []
        self.zshift_history = []
        self.lnlike_smhistory = []
        self.save_threshold = 1e-4
        self.z_prev = np.zeros((self.S,self.M_subspace,self.C_subspace))
        self.F0 = 1.
        self.gof = 1.

    def evaluate_likelihood(self, z):

        x = np.zeros((self.P, self.M, self.C))
        lnL_grad = np.zeros((self.S, self.M_subspace, self.C_subspace))

        lnL, lnL_grad = wavelet_magnitude_colour_position_sparse(z.reshape((self.S, self.M_subspace, self.C_subspace)), self.M, self.C, self.P, *self.wavelet_args, x, lnL_grad)

        return lnL, lnL_grad

    def evaluate_x(self, z):

        from numba.typed import List
        wavelet = List(zip(*self.wavelet_args[4:7]))
        cholesky_m = List(zip(*self.wavelet_args[7:10]))
        cholesky_c = List(zip(*self.wavelet_args[10:13]))

        x = np.zeros((self.P, self.M, self.C))
        x = wavelet_x_sparse(z.reshape((self.S, self.M_subspace, self.C_subspace)), self.M, self.C, *self.wavelet_args[2:4], wavelet, cholesky_m, cholesky_c, x)

        return x

    def merge_likelihoods(self, evaluations):

        lnL=0.
        grad = np.zeros((self.S, self.M_subspace, self.C_subspace))

        for e in evaluations:
            lnL += e[0]
            grad += e[1]

        self.lnlike_iter = lnL
        self.gnorm_iter = np.sqrt(np.sum(grad**2))/(self.S * self.M_subspace * self.C_subspace)
        self.nfev += 1
        self.lnlike_history.append(lnL)

        return lnL, grad.flatten()

    def save_progress(self, X):

        if os.path.exists(self.savefile): mode='a'
        else: mode='w'

        with h5py.File(self.savefile, mode) as hf:
            hf.create_dataset(str(self.nfev), data=X)

    def square_minima(self, X):

        x = np.cumsum(self.zshift_history[-6:][::-1])[::-1]
        y = np.array(self.lnlike_smhistory[-6:])

        M = np.vstack((x**2, x, np.ones(len(x))))
        coeffs = np.linalg.inv(M @ M.T) @ (M @ y)

        self.var = -1/(2*coeffs[0])
        self.mu = -coeffs[1]/(2*coeffs[0])
        self.F0 = coeffs[2] + self.mu**2 / (2*self.var)
        self.gof = (x[-1]-self.mu)**2/self.var

    def fcall(self, X):

        if self.nfev-self.nfev_print>=50:
            self.save_progress(X)
            if self.nfev>20:
                self.fshift = np.abs((self.lnlike_history[-1] - self.lnlike_history[-20])/self.lnlike_history[-1])/20
                self.zshift_history.append(np.sqrt(np.sum((X-self.z_prev)**2)))
                self.lnlike_smhistory.append(self.lnlike_iter)
                if len(self.zshift_history)>3: self.square_minima(X)

            i=0; std_str=""
            for j in self.j:
                if j==-1: std_str+=f"_{np.std(X.reshape(self.S, self.M_subspace, self.C_subspace)[i]):.2f}"
                else: std_str+=f"_{np.std(X.reshape(self.S, self.M_subspace, self.C_subspace)[i:int(i+hp.nside2npix(pow(2,j))+0.1)]):.2f}"
                i+=int(hp.nside2npix(pow(2,j))+0.1)

            print_log(f't={int(time.time()-self.tinit):03d}, n={self.nfev:02d}, lnL={self.lnlike_iter:.0f}, gnorm={self.gnorm_iter:.5f}, fshift={self.fshift:.2e}, std={np.std(X):.3f}{std_str}', logfile=self.logfile)
            #", F0={self.F0:.0f}, gof:{self.gof:.2f}', logfile=self.logfile)

            self.nfev_print = self.nfev

            self.z_prev=X.copy()
            self.lnL_prev=self.lnlike_iter

def evaluate_likelihood(args):
    return args[0].evaluate_likelihood(args[1])
    #return evaluators[iz[0]].evaluate_likelihood(iz[1])
def evaluate_x(iz):
    return iz[0], evaluators[iz[0]].evaluate_x(iz[1])



class pyChisel(Chisel):

    basis_keyword = 'wavelet'

    def evaluate_likelihood(self, z):

        def likelihood(z, S, M, C, P):
            lnL, grad = wavelet_magnitude_colour_position_sparse(z.reshape((S, M, C)), M, C, P, *self.wavelet_args)
            return -lnL, -grad.flatten()

        return likelihood(z.flatten(), self.S, self.M, self.C, self.P)

    def minimize_mp(self, z0, ncores=2, bounds=None, method='BFGS', force=False, nfev_init=0, **scipy_kwargs):

        tstart = time.time()
        from multiprocessing import Pool

        #print('Sigma: ', self.stan_input['sigma'][0], self.stan_input['sigma'][np.cumsum(12 * (2**np.array(self.j[1:]))**2)[:-1] - 1])
        sigma = np.unique(self.stan_input['sigma'], return_index=True)
        print('Sigma: ', sigma[0][np.argsort(sigma[1])])

        logfile = os.path.join(self.output_directory,self.file_root+'_log.txt')
        savefile = os.path.join(self.output_directory,self.file_root+'_progress.h')
        if os.path.exists(savefile):
            if not force: raise OSError(f"File {savefile} already exists, won't overwrite.")

        print('Initialising arguments.')
        self._generate_args(sparse=True)
        self._generate_args_ray(nsets=ncores, sparse=True)

        print('Initialising multiprocessing processes.')
        global evaluators
        evaluators = [evaluate(self.P_ray[i], self.S,
                               self.M, self.C, self.M_subspace, self.C_subspace,
                               self.wavelet_args_ray[i], j=self.j,
                               logfile=logfile, savefile=savefile) for i in range(ncores)]
        evaluators[0].nfev=nfev_init

        self.optimum_results_file = self.file_root+'_scipy_results.h5'

        with Pool(ncores) as pool:
            icore = np.arange(ncores)

            def likelihood(z):
                evaluators[0].z = z
                evaluations = pool.map(evaluate_likelihood, zip(evaluators, np.repeat([z,],ncores, axis=0)))
                lnL, lnL_grad =  evaluators[0].merge_likelihoods(evaluations)

                if evaluators[0].fshift<evaluators[0].save_threshold:
                    evaluators[0].save_threshold *= 0.1
                    print('Processing...')
                    self.optimum_lnp = lnL - 0.5*np.sum(z**2)
                    self.optimum_lnp_history = np.array(evaluators[0].lnlike_history)
                    self.optimum_z = z.reshape((self.S, self.M_subspace, self.C_subspace))
                    self.optimum_b, self.optimum_x = self._get_bx(self.optimum_z)
                    if self.nest: self.optimum_x = self._ring_to_nest(np.moveaxis(self.optimum_x, 0, -1))
                    self.save_h5(time.time()-tstart, 'optimum_lnp_history')


                return -lnL + 0.5*np.sum(z**2), -lnL_grad.flatten() + z
            callback=evaluators[0].fcall

            global tinit; tinit = time.time()

            print('z0 likelihood')
            likelihood(z0); evaluators[0].fcall(z0)
            print('Running optimizer.')
            res = scipy.optimize.minimize(likelihood, z0.flatten(), method=method, jac=True, bounds=bounds, callback=callback, **scipy_kwargs)

            self.optimum_lnp_history = np.array(evaluators[0].lnlike_history)

        print('Processing results.')
        self.optimum_lnp = -res['fun']
        self.optimum_z = res['x'].reshape((self.S, self.M_subspace, self.C_subspace))
        self.optimum_b, self.optimum_x = self._get_bx(self.optimum_z)
        if self.nest: self.optimum_x = self._ring_to_nest(np.moveaxis(self.optimum_x, 0, -1))

        self.save_h5(time.time()-tstart, 'optimum_lnp_history')

        print_log(str(res), logfile=logfile)

        return res

    def minimize(self, z0, ncores=2, bounds=None, method='BFGS', **scipy_kwargs):

        tstart = time.time()

        print('Initialising arguments.')
        self._generate_args(sparse=True)

        def likelihood(z):
            x = np.zeros((self.P, self.M, self.C))
            lnL_grad = np.zeros((self.S, self.M_subspace, self.C_subspace))
            lnL, grad = wavelet_magnitude_colour_position_sparse(z.reshape((self.S, self.M_subspace, self.C_subspace)), self.M, self.C, self.P, *self.wavelet_args, x, lnL_grad)
            global lnlike_iter; lnlike_iter = lnL
            global gnorm_iter; gnorm_iter = np.sum(np.abs(grad))
            return -lnL + 0.5*np.sum(z**2), -grad.flatten() + z

        global tinit
        tinit = time.time()

        res = scipy.optimize.minimize(likelihood, z0.flatten(), method=method, jac=True, bounds=bounds, callback=fcall, **scipy_kwargs)

        print('Processing results.')
        self.optimum_lnp = -res['fun']
        self.optimum_z = res['x'].reshape((self.S, self.M_subspace, self.C_subspace))
        self.optimum_b, self.optimum_x = self._get_bx(self.optimum_z)
        if self.nest: self.optimum_x = self._ring_to_nest(np.moveaxis(self.optimum_x, 0, -1))

        # Save optimum to h5py
        self.optimum_results_file = self.file_root+'_scipy_results.h5'
        self.save_h5(time.time()-tstart)

        return res

    def _evaluate_likelihood(self, z, ncores=1, generate=True, iset=-1):

        if generate:
            # numba.set_num_threads(ncores)
            self._generate_args(sparse=True)

        lnL_grad = np.zeros((self.S, self.M_subspace, self.C_subspace))
        if iset==-1:
            x = np.zeros((self.P, self.M, self.C))
            lnL, grad = wavelet_magnitude_colour_position_sparse(z.reshape((self.S, self.M_subspace, self.C_subspace)), self.M, self.C, self.P, *self.wavelet_args, x, lnL_grad)
        else:
            x = np.zeros((self.P_ray[iset], self.M, self.C))
            lnL, grad = wavelet_magnitude_colour_position_sparse(z.reshape((self.S, self.M_subspace, self.C_subspace)), self.M, self.C, self.P_ray[iset], *self.wavelet_args_ray[iset], x, lnL_grad)

        return lnL, grad

    def _get_bx(self, z):

        from numba.typed import List
        print('Setting cholesky arguments')
        wavelet = List(zip(*self.wavelet_args[4:7]))
        cholesky_m = List(zip(*self.wavelet_args[7:10]))
        cholesky_c = List(zip(*self.wavelet_args[10:13]))

        print('Get b')
        b = np.zeros((self.S, self.M, self.C))
        b = wavelet_b_sparse(z.reshape((self.S, self.M_subspace, self.C_subspace)), self.M, self.C, self.S, *self.wavelet_args[2:4], cholesky_m, cholesky_c, b)
        print('Get x')
        x = np.zeros((self.P, self.M, self.C))
        x = wavelet_x_sparse(z.reshape((self.S, self.M_subspace, self.C_subspace)), self.M, self.C, *self.wavelet_args[2:4], wavelet, cholesky_m, cholesky_c, x)

        return b, x

    def _get_bx_mp(self, z, ncores=2):

        tstart = time.time()
        from multiprocessing import Pool
        from numba.typed import List

        print('Initialising arguments.')
        self._generate_args(sparse=True)
        self._generate_args_ray(nsets=ncores, sparse=True)

        print('Setting cholesky arguments')
        cholesky_m = List(zip(*self.wavelet_args[7:10]))
        cholesky_c = List(zip(*self.wavelet_args[10:13]))

        print('Getting b')
        b = np.zeros((self.S, self.M, self.C))
        b = wavelet_b_sparse(z.reshape((self.S, self.M_subspace, self.C_subspace)), self.M, self.C, self.S, *self.wavelet_args[2:4], cholesky_m, cholesky_c, b)
        self.optimum_b = b.copy()

        print('Initialising multiprocessing processes.')
        global evaluators
        evaluators = [evaluate(self.P_ray[i], self.S, self.M, self.C, self.M_subspace, self.C_subspace, self.wavelet_args_ray[i], logfile="Bad_directory", savefile="Bad_directory") for i in range(ncores)]

        icore = np.arange(ncores)
        with Pool(ncores) as pool:
            evaluations = pool.map(evaluate_x, zip(icore, np.repeat([z,],ncores, axis=0)))

        self.optimum_x = np.concatenate([evaluations[ii][1] for ii in icore])
        print([evaluations[ii][0] for ii in icore])
        if self.nest: self.optimum_x = self._ring_to_nest(np.moveaxis(self.optimum_x, 0, -1))


    def _cholesky_args(self, sparse=False):

        if sparse:
            self.cholesky_m = scipy.sparse.csr_matrix((self.stan_input['cholesky_w_m'],
                                                self.stan_input['cholesky_v_m']-1,
                                                self.stan_input['cholesky_u_m']-1)).toarray()
            self.cholesky_c = scipy.sparse.csr_matrix((self.stan_input['cholesky_w_c'],
                                            self.stan_input['cholesky_v_c']-1,
                                            self.stan_input['cholesky_u_c']-1)).toarray()

            cholesky_u_m = np.zeros(len(self.stan_input['cholesky_v_m']), dtype=int)
            for iS, iY in enumerate(self.stan_input['cholesky_u_m'][1:]-1):
                cholesky_u_m[iY:] += 1
            cholesky_u_c = np.zeros(len(self.stan_input['cholesky_v_c']), dtype=int)
            for iS, iY in enumerate(self.stan_input['cholesky_u_c'][1:]-1):
                cholesky_u_c[iY:] += 1

            cholesky_args = [cholesky_u_m,
                             self.stan_input['cholesky_v_m']-1,
                             self.stan_input['cholesky_w_m'],
                             cholesky_u_c,
                             self.stan_input['cholesky_v_c']-1,
                             self.stan_input['cholesky_w_c']]

            self.wavelet_model = wavelet_magnitude_colour_position_sparse
        elif not sparse:
            self.cholesky_m = scipy.sparse.csr_matrix((self.stan_input['cholesky_w_m'],
                                                self.stan_input['cholesky_v_m']-1,
                                                self.stan_input['cholesky_u_m']-1)).toarray()
            self.cholesky_c = scipy.sparse.csr_matrix((self.stan_input['cholesky_w_c'],
                                            self.stan_input['cholesky_v_c']-1,
                                            self.stan_input['cholesky_u_c']-1)).toarray()
            cholesky_args = [self.cholesky_m, self.cholesky_c]

        return cholesky_args

    def _generate_args(self, sparse=False):

        cholesky_args = self._cholesky_args(sparse=sparse)

        self.wavelet_model = wavelet_magnitude_colour_position

        lnL_grad = np.zeros((self.S, self.M, self.C))
        x = np.zeros((self.M, self.C))

        self.wavelet_args = [np.moveaxis(self.k, -1,0).astype(np.int64).copy(),np.moveaxis(self.n, -1,0).astype(np.int64).copy()] \
                          + [self.stan_input[arg].copy() for arg in ['mu', 'sigma']]\
                          + [self.stan_input['wavelet_U'].copy()-1,
                             self.stan_input['wavelet_v'].copy()-1,
                             self.stan_input['wavelet_w'].copy()]\
                          + cholesky_args

    def _generate_args_ray(self, nsets=1, sparse=False):

        cholesky_args = self._cholesky_args(sparse=sparse)

        P_ray = np.zeros(nsets, dtype=int) + self.P//nsets
        P_ray[:self.P - np.sum(P_ray)] += 1
        print('P sets: ', P_ray, np.sum(P_ray))

        wavelet_u = self.stan_input['wavelet_U'].copy()-1
        wavelet_v = self.stan_input['wavelet_v'].copy()-1
        wavelet_w = self.stan_input['wavelet_w'].copy()

        self.wavelet_args_ray = []
        iP = 0
        for iset in range(nsets):

            lnL_grad = np.zeros((self.S, self.M, self.C))
            x = np.zeros((self.M, self.C))

            wavelet_args_set  = [np.moveaxis(self.k, -1,0).astype(np.int64)[iP:iP+P_ray[iset]].copy(),
                                 np.moveaxis(self.n, -1,0).astype(np.int64)[iP:iP+P_ray[iset]].copy()] \
                              + [self.stan_input[arg].copy() for arg in ['mu', 'sigma']] \
                              + [arg[int(self.stan_input['wavelet_u'][iP]-1):int(self.stan_input['wavelet_u'][iP+P_ray[iset]]-1)].copy() \
                                                                        for arg in [wavelet_u,wavelet_v,wavelet_w]] \
                              + cholesky_args
            wavelet_args_set[4]-=iP

            self.wavelet_args_ray.append(wavelet_args_set)
            iP += P_ray[iset]

        self.P_ray = P_ray
