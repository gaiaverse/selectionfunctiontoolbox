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


####%%%% Original functions kept here just in case needed in future - can be deleted later!

@njit(parallel=parallel)
def wavelet_x_sparse_OG(z, M, C, P, mu, sigma,
            wavelet_u, wavelet_v, wavelet_w,
            cholesky_u_m, cholesky_v_m, cholesky_w_m,
            cholesky_u_c, cholesky_v_c, cholesky_w_c,
            x, MC):

    #x = np.zeros((P, M, C))
    #MC = np.zeros((M, C))
    x *= 0.
    MC *= 0.

    # Iterate over pixels
    iY = 0
    for ipix in prange(P):

        # Iterate over modes which are not sparsified in Y
        iSmin = wavelet_u[ipix]
        iSmax = wavelet_u[ipix+1]
        for iS in wavelet_v[iSmin:iSmax]:
            MC *= 0.
            iYmag = 0
            for iM in prange(M):
                iMmin = cholesky_u_m[iM]
                iMmax = cholesky_u_m[iM+1]
                for iMsub in cholesky_v_m[iMmin:iMmax]:
                    iYcol = 0
                    for iC in prange(C):
                        iCmin = cholesky_u_c[iC]
                        iCmax = cholesky_u_c[iC+1]
                        for iCsub in cholesky_v_c[iCmin:iCmax]:
                            MC[iM, iC] += (cholesky_w_m[iYmag] * z[iS,iMsub,iCsub] * cholesky_w_c[iYcol]);

                            iYcol+=1
                    iYmag += 1

            x[ipix] += (mu[iS] + sigma[iS]*MC)* wavelet_w[iY]

            iY += 1

    return x

@njit
def wavelet_b_sparse_OG(z, M, C, S, mu, sigma,
            wavelet_u, wavelet_v, wavelet_w,
            cholesky_u_m, cholesky_v_m, cholesky_w_m,
            cholesky_u_c, cholesky_v_c, cholesky_w_c,
            b, MC):

    #x = np.zeros((P, M, C))
    #MC = np.zeros((M, C))
    b *= 0.
    MC *= 0.

    # Iterate over modes which are not sparsified in Y
    for iS in range(S):
        MC *= 0.
        iYmag = 0
        for iM in range(M):
            iMmin = cholesky_u_m[iM]
            iMmax = cholesky_u_m[iM+1]
            for iMsub in cholesky_v_m[iMmin:iMmax]:
                iYcol = 0
                for iC in range(C):
                    iCmin = cholesky_u_c[iC]
                    iCmax = cholesky_u_c[iC+1]
                    for iCsub in cholesky_v_c[iCmin:iCmax]:
                        MC[iM, iC] += (cholesky_w_m[iYmag] * z[iS,iMsub,iCsub] * cholesky_w_c[iYcol]);

                        iYcol+=1
                iYmag += 1

        b[iS] = (mu[iS] + sigma[iS]*MC)

    return b

@njit(parallel=parallel)
def wavelet_magnitude_colour_position_sparse_OG(z, M, C, P, k, n, mu, sigma,
                                                wavelet_u, wavelet_v, wavelet_w,
                                                cholesky_u_m, cholesky_v_m, cholesky_w_m,
                                                cholesky_u_c, cholesky_v_c, cholesky_w_c,
                                                x, lnL_grad, MC):

    # x = np.zeros((M, C))
    # lnL_grad = np.zeros((S, M_subspace, C_subspace))

    x = wavelet_x_sparse_OG(z, M, C, P, mu, sigma,
                wavelet_u, wavelet_v, wavelet_w,
                cholesky_u_m, cholesky_v_m, cholesky_w_m,
                cholesky_u_c, cholesky_v_c, cholesky_w_c,
                x, MC)

    lnL = 0.
    lnL_grad *= 0.

    # Iterate over pixels
    iY = 0
    for ipix in prange(P):

        d = 1 + np.exp(-np.abs(x[ipix]))

        # Iterate over modes which are not sparsified in Y
        iSmin = wavelet_u[ipix]
        iSmax = wavelet_u[ipix+1]
        # Likelihood gradient db/dz * dx/db * dlnL/dx
        for iS in wavelet_v[iSmin:iSmax]:
            # Evaluate b from z at iS mode
            iYmag = 0
            for iM in prange(M):
                iMmin = cholesky_u_m[iM]
                iMmax = cholesky_u_m[iM+1]
                for iMsub in cholesky_v_m[iMmin:iMmax]:
                    iYcol = 0
                    for iC in prange(C):

                        iCmin = cholesky_u_c[iC]
                        iCmax = cholesky_u_c[iC+1]
                        for iCsub in cholesky_v_c[iCmin:iCmax]:

                            lnL_grad[iS,iMsub,iCsub] += sigma[iS] * cholesky_w_m[iYmag] * cholesky_w_c[iYcol] * wavelet_w[iY] \
                                          * (k[ipix,iM,iC] - n[ipix,iM,iC]*(0.5 + np.sign(x[ipix,iM,iC])*(0.5+(1-d[iM,iC])/d[iM,iC])) )

                            iYcol+=1
                    iYmag+=1


            iY += 1

        #lnL += np.sum( -k[ipix]*np.log(1+1/exp_x) - (n[ipix]-k[ipix])*np.log(1+exp_x) )
        #lnL += np.sum( k[ipix]*x - n[ipix]*np.log1p(exp_x) )
        #lnL += np.sum( k[ipix]*x - n[ipix]*(x/2 + np.log(2*np.cosh(x/2)) ) )
        lnL += np.sum( k[ipix]*x[ipix] - n[ipix]*(x[ipix]/2 + np.abs(x[ipix])/2 + np.log(d) ) )

    # Add on Gaussian prior
    return lnL - 0.5*np.sum(z**2), lnL_grad - z

@njit
def wavelet_magnitude_colour_position(z, M, C, P, k, n, mu, sigma,
                                     wavelet_u, wavelet_v, wavelet_w, cholesky_m, cholesky_c,
                                     lnL_grad, x):

    lnL = 0.
    lnL_grad *= 0.
    lnL_grad_local = np.zeros(lnL_grad.shape)

    b = np.zeros(z.shape[1:])

    # Iterate over pixels
    for ipix in range(P):
        x *= 0.
        lnL_grad_local *= 0.

        # Iterate over modes which are not sparsified in Y
        imin = wavelet_u[ipix]
        imax = wavelet_u[ipix+1]
        iY = 0
        for iS in wavelet_v[imin:imax]:

            # Evaluate b from z at iS mode
            b = mu[iS] + sigma[iS] * (cholesky_m @ z[iS] @ cholesky_c.T);

            # Evaluate x from b
            x += b * wavelet_w[int(wavelet_u[ipix]+iY)]
            #x += b * wavelet_w[wavelet_u[ipix]+iY]

            # Likelihood gradient - dx/db
            lnL_grad_local[iS] += wavelet_w[int(wavelet_u[ipix]+iY)]

            iY += 1

        exp_x = np.exp(x)

        # Likelihood gradient db/dz * dx/db * dlnL/dx
        for iS in wavelet_v[imin:imax]:
            lnL_grad[iS] += \
            sigma[iS] * cholesky_m.T @ ( lnL_grad_local[iS] * (k[ipix] - n[ipix]/(1+1/exp_x)) ) @ cholesky_c
            #lnL_grad[iS] += lnL_grad_local[iS] * (k[ipix] - n[ipix]/(1+1/exp_x))

        #lnL += np.sum( -k[ipix]*np.log(1+1/exp_x) - (n[ipix]-k[ipix])*np.log(1+exp_x) )
        lnL += np.sum( k[ipix]*x - n[ipix]*np.log1p(exp_x) )

        #(k*x - (n-k))*(x/2)*log(e^x/2 + e^-x/2)

    # Add on Gaussian prior
    return lnL + np.sum(-0.5*z**2), lnL_grad - z


@njit(parallel=parallel)
def wavelet_x_sparse4(z, MMs, CCs, SP, mu, sigma,
            wavelet_u, wavelet_v, wavelet_w,
            cholesky_u_m, cholesky_v_m, cholesky_w_m,
            cholesky_u_c, cholesky_v_c, cholesky_w_c, x):

    #x = np.zeros((P, M, C))
    x *= 0.
    for iY in range(SP):
        for iYmag in range(MMs):
            for iYcol in range(CCs):
                x[wavelet_u[iY],cholesky_u_m[iYmag],cholesky_u_c[iYcol]] += cholesky_w_m[iYmag] * z[wavelet_v[iY],cholesky_v_m[iYmag],cholesky_v_c[iYcol]] * cholesky_w_c[iYcol] * sigma[wavelet_v[iY]] * wavelet_w[iY]
                #x[ipix,iM,iC] += mu[iS] * wavelet_w[iY]

    return x
