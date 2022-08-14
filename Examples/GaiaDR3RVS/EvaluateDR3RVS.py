import sys, os, pickle, time, warnings
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import h5py, numpy as np, scipy.stats, healpy as hp, tqdm

if __name__=='__main__':

    output_dir, count_file, M, C, jmax, mag_res, col_res, healpix = sys.argv[1:]
    M=int(M); C=int(C); jmax=int(jmax);
    # Gmin=float(Gmin); Cmin=float(Cmin)
    # Gmax=float(Gmax); Cmax=float(Cmax)
    mag_res=float(mag_res); col_res=float(col_res); healpix=int(healpix)

    eps=1e-10

    ### Parameters to update for different runs ###
    ncores=6
    Mlims = [5.,17]; Clims = [-1,3];
    nside=pow(2,healpix); B=2.

    length_m = 0.3; length_c = 300
    sigma = [-0.0755305 , -2.24756519]

    output_dir = '/Users/adminaeverall/Documents/Astro/Data/GaiaDR3RVS_run'

    ###  You shouldn't need to change anything beyond this point  ###
    colour=True
    M_bins = np.arange(Mlims[0], Mlims[1]+eps, mag_res)
    C_bins = np.arange(Clims[0], Clims[1]+eps, col_res)

    data_M=int((Mlims[1]-Mlims[0])/mag_res + eps);
    data_C=int((Clims[1]-Clims[0])/col_res + eps);
    data_nside = pow(2,healpix)
    data_res=(data_M, data_C, hp.nside2npix(data_nside))
    print('data_res: ', data_res)

    box={};
    with h5py.File(count_file, 'r') as hf:
        box['n'] = np.zeros(data_res, dtype=np.int64)
        box['k'] = np.zeros(data_res, dtype=np.int64)

        Midx = hf['magnitude'][...] - int(Mlims[0]/mag_res + eps)
        try: Cidx = hf['colour'][...] - int(Clims[0]/col_res + eps)
        except KeyError: Cidx = np.zeros(len(Midx), dtype=np.int64)
        Pidx = hf['position'][...]
        in_range = (Midx>-1)&(Midx<data_M)&(Cidx>-1)&(Cidx<data_C)
        print(np.unique(hf['magnitude'][...]), np.unique(hf['colour'][...]))
        for key in ['n','k']:
            box[key][Midx[in_range], Cidx[in_range], Pidx[in_range]] = hf[key][...][in_range]
    print(box['n'].shape)


    # Calculate lengthscales in units of bins
    M_original, C_original = box['k'].shape[:2]
    lengthscale_m = length_m/((M_bins[1]-M_bins[0])*(M_original/M))
    lengthscale_c = length_c/((C_bins[1]-C_bins[0])*(C_original/C))
    print(f"lengthscales m:{lengthscale_m} , c:{lengthscale_c}")

    file_root = f"chisquare_rvs_jmax{jmax}_nside{nside}_M{M}_CGR{C}_lm{length_m}lc{length_c}_B{B}"
    print(file_root)
    basis_options = {'needlet':'chisquare', 'j':jmax, 'B':B, 'p':1.0, 'wavelet_tol':1e-2}

    # Import chisel
    sys.path.append("PythonScripts/")
    from SelectionFunctionPython import pyChisel
    pychisel = pyChisel(box['k'], box['n'],
                    basis_options,file_root,
                    axes = ['magnitude','colour','position'],
                    nest = True,
                    lengthscale_m = lengthscale_m,
                    lengthscale_c = lengthscale_c,
                    M = M,
                    C = C,
                    nside = nside,
                    sparse = True,
                    pivot = True,
                    mu = 0.0,
                    sigma = sigma,
                    Mlim = [M_bins[0], M_bins[-1]],
                    Clim = [C_bins[0], C_bins[-1]],
                    spherical_basis_directory=os.path.join(output_dir,'SphericalWavelets'),
                    output_directory=os.path.join(output_dir,'PyOutput')
                    )
    print('j', pychisel.j)

    if True:
        z0 = np.random.normal(0, 1, size=(pychisel.S, pychisel.M_subspace, pychisel.C_subspace)).flatten()
        last_iteration=0
        force=False
    else:
        print('Hot Start!')
        with h5py.File(os.path.join(output_dir,f'PyOutput/{file_root}_progress.h'), 'r') as hf:
            keys = list(hf.keys())
            z0 = hf[keys[np.argmax(np.array(keys).astype(int))]][...]
            last_iteration = np.max(np.array(keys).astype(int))
        force=True

    f_tol = 1e-10
    print(f'f_tol = {f_tol:.0e}')
    bounds=np.zeros((len(z0.flatten()), 2)); bounds[:,0]=-50; bounds[:,1]=50
    res = pychisel.minimize_mp(z0, ncores=ncores, bounds=bounds, method='L-BFGS-B', force=force, nfev_init=last_iteration,
                                   options={'disp':False, 'maxiter':25000, 'maxfun':25000, 'ftol':f_tol, 'gtol':1})
    print(res)
