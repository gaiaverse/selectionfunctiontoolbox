import numpy as np
from numba import njit, jit, prange, config

config.THREADING_LAYER = 'threadsafe'
parallel = False

eps=1e-10

@jit(parallel=parallel)
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
