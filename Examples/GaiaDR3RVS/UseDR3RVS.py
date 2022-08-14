import sys, os
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
import matplotlib.pyplot as plt
from pylab import cm
from matplotlib.colors import LogNorm
import matplotlib

from selectionfunctions.source import Source

plt.rc('axes', labelsize=24)
plt.rc('xtick',labelsize=24)
plt.rc('ytick',labelsize=24)
plt.rc('legend',fontsize=24)

import numpy as np, h5py, healpy as hp, pandas as pd
import scipy.stats

from selectionfunctions.config import config
config['data_dir'] = '/Users/adminaeverall/Documents/Astro/Data/fitted_selectionfunctions/'
import selectionfunctions.cog_v as CoGV

# Evaluate SF prob
import astropy.units as units


logit = lambda q: np.log(q/(1-q))
expit = lambda x: np.exp(x)/(1+np.exp(x))


if __name__=='__main__':

    output_dir, map_fname = sys.argv[1:]
    config['data_dir'] = os.path.join(output_dir, 'PyOutput')
    rvs_sf = CoGV.subset_sf(map_fname=map_fname, nside=pow(2,4),
                    basis_options={'needlet':'chisquare', 'p':1.0, 'wavelet_tol':1e-2},
                    spherical_basis_directory='/Users/adminaeverall/Documents/Astro/Data/GaiaDR3RVS_run/SphericalWavelets/')



    g = [15.5]
    c = [1.]

    nside=2**4
    ra, dec = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), nest=True, lonlat=True)
    coords = Source(np.repeat(ra,len(g)*len(c),axis=0).reshape(-1,len(g),len(c))*units.deg,
                    np.repeat(dec,len(g)*len(c),axis=0).reshape(-1,len(g),len(c))*units.deg,
                    frame='icrs',
                    photometry={'gaia_g':np.moveaxis(np.repeat(g,len(ra)*len(c),axis=0).reshape(-1,len(ra),len(c)),0,1),
                                'gaia_g_gaia_rp':np.moveaxis(np.repeat(c,len(ra)*len(g),axis=0).reshape(-1,len(ra),len(c)),0,2),})

    probability = rvs_sf(coords)

    minmax=[np.log(1/999),np.log(0.999/0.001)]
    hp.mollview(logit(probability[:,0,0]), nest=True, notext=True,min=minmax[0],max=minmax[1],
                coord=['C','G'], title='', cmap='PRGn', hold=True, cbar=False, xsize=800)

    print(os.path.join(f"{output_dir}",f"G{g[0]}C{c[0]}_RVSSF_logit.png"))
    plt.savefig(os.path.join(f"{output_dir}",f"G{g[0]}C{c[0]}_RVSSF_logit.png"),
                bbox_inches='tight', dpi=200, facecolor='w', transparent=False)
