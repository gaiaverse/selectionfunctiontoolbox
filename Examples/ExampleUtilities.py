import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from scipy import special, stats
from matplotlib.colors import Normalize
import matplotlib.ticker

def magnitude_selection_plot(filename,magnitude_bins,n,k,x,magnitude_lines=[]):

    plt.stairs(n.sum(axis=1),edges=magnitude_bins,label='2MASS')
    plt.stairs(k.sum(axis=1),edges=magnitude_bins,label='APOGEE')
    
    rate = (n*special.expit(x[:,0,:])).sum(axis=1)
    percentiles = stats.poisson(mu=rate[:,np.newaxis]).ppf(np.array([[0.05,0.16,0.50,0.84,0.95]]))
    plt.bar(magnitude_bins[:-1], height=percentiles[:,4]-percentiles[:,0], width=magnitude_bins[1]-magnitude_bins[0], bottom=percentiles[:,0], align='edge', alpha=0.3, color='ForestGreen')
    plt.bar(magnitude_bins[:-1], height=percentiles[:,3]-percentiles[:,1], width=magnitude_bins[1]-magnitude_bins[0], bottom=percentiles[:,1], align='edge', alpha=0.3, color='ForestGreen')
    plt.stairs(percentiles[:,2],edges=magnitude_bins,label='Prediction',color='ForestGreen')
    
    for _lines in magnitude_lines:
        plt.plot([_lines,_lines],[5e-1,5e8],lw=1,color='lightgrey',ls='--',zorder=0)
    plt.yscale('log')
    plt.xlabel(r'$H\;(\mathrm{mag})$',fontsize=14)
    plt.ylabel(r'Number of sources',fontsize=14)
    plt.legend(frameon=False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    # Ensure logarithmic ticks appear on y axis
    locmaj = matplotlib.ticker.LogLocator(base=10,numticks=12) 
    plt.gca().yaxis.set_major_locator(locmaj)
    locmin = matplotlib.ticker.LogLocator(base=10.0,subs=(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),numticks=12)
    plt.gca().yaxis.set_minor_locator(locmin)
    plt.gca().yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    plt.ylim([5e-1,5e8])
    plt.xlim([-5,25])
    plt.savefig(f'./ExampleResults/{filename}.pdf',dpi=300,bbox_inches='tight')
    plt.savefig(f'./ExampleResults/{filename}.png',dpi=300,bbox_inches='tight',facecolor='w')

def position_selection_plot(filename,magnitudes_to_plot,magnitude_bins,n,k,x,mask=None):
    fig, axes = plt.subplots(3,4,figsize=(12,4.75))
    fig.subplots_adjust(hspace=0.1,wspace=0.0)
    
    bins_to_plot = [np.sum(magnitude_bins<magnitudes_to_plot[i])-1 for i in range(4)]

    p=np.array([0.001,0.01,0.1,0.5,0.9,0.99,0.999]); ticks=np.log(p/(1-p))
    tick_labels = [rf"${ticks[i]:.1f}\,({p[i]*100:.1f}\%)$" for i in range(len(p))]

    probability_kwargs = {'nest':True,
                       'notext':True,
                       'min':special.logit(0.001),
                       'max':special.logit(0.999),
                       'coord':['C','G'],
                       'cmap':'PRGn',
                       'badcolor':'grey',
                       'hold':True,
                       'cbar':False, 
                       'xsize':2000}
    
    mask_kwargs = probability_kwargs.copy()
    mask_kwargs['hold'] = False
    mask_kwargs['reuse_axes'] = True
    
    consistency_kwargs = probability_kwargs.copy()
    consistency_kwargs['min'] = 0
    consistency_kwargs['max'] = 1
    consistency_kwargs['cmap'] = 'RdBu_r'

    for i in range(4):

        mag_idx = bins_to_plot[i]

        plt.sca(axes[0,i])
        estimated_selection_probability = (k[mag_idx]+1)/(n[mag_idx]+2)
        hp.mollview(special.logit(estimated_selection_probability), title='H=%.1f'%magnitudes_to_plot[i], **probability_kwargs)
        if mask is not None: hp.mollview(mask*np.nan,alpha=mask, title='H=%.1f'%magnitudes_to_plot[i],**mask_kwargs)
        hp.graticule(alpha=0.0)
        if i==0: plt.text(-2.5,0.,'Data',ha='center', va='center', rotation='vertical',fontsize=14)

        plt.sca(axes[1,i])
        hp.mollview(x[mag_idx,0], title='', **probability_kwargs)
        if mask is not None: hp.mollview(mask*np.nan,alpha=mask, title='',**mask_kwargs)
        hp.graticule(alpha=0.0)
        if i==0: plt.text(-2.5,0.,'Model',ha='center', va='center', rotation='vertical',fontsize=14)

        plt.sca(axes[2,i])
        pval_k = stats.binom.cdf(k[mag_idx], n[mag_idx], special.expit(x[mag_idx,0]))
        pval_km1 = stats.binom.cdf(k[mag_idx]-1, n[mag_idx], special.expit(x[mag_idx,0]))
        pvals = np.random.rand(len(pval_k))*(pval_k - pval_km1) + pval_km1
        hp.mollview(pvals, title='', **consistency_kwargs)
        if mask is not None: hp.mollview(mask*np.nan,alpha=mask, title = '',**mask_kwargs)
        hp.graticule(alpha=0.0)
        if i==0: plt.text(-2.5,0.,'Consistency',ha='center', va='center', rotation='vertical',fontsize=14)



    ax = fig.add_axes([0.21, 0.95, 0.6, 0.03]); 
    norm = Normalize(vmin=probability_kwargs['min'], vmax=probability_kwargs['max'])
    im = plt.cm.ScalarMappable(norm=norm, cmap=probability_kwargs['cmap']); im.set_array([])
    cbar = plt.colorbar(im, cax=ax, orientation='horizontal', ticks=ticks)
    cbar.set_ticklabels(tick_labels)
    cbar.set_label(r'$x = \mathrm{logit}(p)$',fontsize=14,labelpad = 6)
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')


    ax = fig.add_axes([0.21, 0.05, 0.6, 0.03]); 
    norm = Normalize(vmin=consistency_kwargs['min'], vmax=consistency_kwargs['max'])
    im = plt.cm.ScalarMappable(norm=norm, cmap=consistency_kwargs['cmap']); im.set_array([])
    consistency_ticks = np.arange(0,1.01,0.1)
    consistency_ticklabels = [rf"${consistency_ticks[i]*100:.0f}\%$" for i in range(len(consistency_ticks))]
    cbar = plt.colorbar(im, cax=ax, orientation='horizontal', ticks=consistency_ticks)
    cbar.set_ticklabels(consistency_ticklabels)
    cbar.set_label(r'$p_\mathrm{value}$', fontsize=14,labelpad = -2)

    plt.savefig(f'./ExampleResults/{filename}.pdf',dpi=300,bbox_inches='tight')
    plt.savefig(f'./ExampleResults/{filename}.png',dpi=300,bbox_inches='tight',facecolor='w')