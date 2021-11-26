from __future__ import division
import numpy as np
import matplotlib
import matplotlib.pyplot as pp
import pyregion
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.transforms import Affine2D
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter
import pandas as pd

from astropy.visualization import AsinhStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from matplotlib import rc
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec

rc('font', family='serif')
matplotlib.rcParams['font.size'] = (8)

_true_pat = 'data/sci_mock_JWST_{}_{}_onlyHost.fits'
_mag_zp = 25.9463

_stretch = AsinhStretch()
#LW stretch
#_stretch.a = (0.05 - 0.0001)/2 / (0.05+0.0001)
#_pnorm = ImageNormalize(vmin=-0.0001, vmax=0.05, stretch=_stretch, clip=True)
_stretch.a = (0.01 - 0.0001)/2 / (0.01+0.0001)
_pnorm = ImageNormalize(vmin=-0.0001, vmax=0.01, stretch=_stretch, clip=True)
_axis_range = [-0.6,0.6,-0.6,0.6]#[-2.5, 2.5, -2.5, 2.5]  # in arcsec
#_xytix = [-3,-2, -1, 0, 1, 2,3]  # in arcsec
_xytix = [-0.5, 0, 0.5]  # in arcsec
_coltix = np.array([27,28,29])  # in mag/arcsec**2

gray_r = pp.cm.cmap_d['Spectral_r']


def mag_to_flux(mag, zp=0.0, scale=(1.0, 1.0)):
    return 10**(-0.4*(mag - zp)) * np.prod(scale)


def plot_models(quasar,filt):
    if filt =='F200W':
      _psfresid_pat = 'runJWST/SDSS_z7_SN/mcmc_out_mock_JWST_{}_point_source_subtracted.fits'
      psfresid = fits.getdata(_psfresid_pat.format(quasar))
    else:
      _psfresid_pat = 'runJWST/SDSS_z7_{}/mcmc_out_mock_JWST_{}_point_source_subtracted.fits'
      psfresid = fits.getdata(_psfresid_pat.format(filt,quasar))
    #psfresid_smooth = gaussian_filter(psfresid, (2, 2))
    resid_smooth = gaussian_filter(psfresid, (1, 1))
    #resid_smooth = psfresid

    true = fits.getdata(_true_pat.format(filt,quasar))
    #psfresid_smooth = gaussian_filter(psfresid, (2, 2))
    true_smooth = gaussian_filter(true, (1, 1))
    #true_smooth = true

    center = np.array(psfresid.shape)[::-1]/2
    pxscale = 0.031/2
    extents = np.array([-center[0], center[0],
               -center[1], center[1]])*pxscale

    #plot_panels = [psfresid, 'Point Source\nSubtracted']
    plot_panels = [resid_smooth, 'Point Source\nSubtracted']

    im = grid[ii].imshow(plot_panels[0], extent=extents, origin='lower',
                               cmap=gray_r, norm=_pnorm,
                               interpolation='nearest')
    im = grid[ii+1].imshow(true_smooth, extent=extents, origin='lower',
                               cmap=gray_r, norm=_pnorm,
                               interpolation='nearest')
    
    grid[ii].axis(_axis_range)
    ticks = mag_to_flux(_coltix, zp=_mag_zp, scale=pxscale)
    cbar = pp.colorbar(im, cax=grid.cbar_axes[0], ticks=ticks)
    cbar.set_ticklabels(_coltix)
    grid.cbar_axes[0].set_ylabel('mag arcsec$^{-2}$')
    grid.cbar_axes[0].set_xlabel('mag arcsec$^{-2}$')
    grid.cbar_axes[0].yaxis.set_label_coords(2,0.5)

    #grid[ii].set_title(quasar)

if __name__ == '__main__':
    from sys import argv
 
    if len(argv)>1:
      filt=str(argv[1])
    else:
      filt='F200W'


    # import glob
    to_plot = [2,   3,   6,   7,   8,   9,  10,  12,  16, 18,  20,  22,  23,  25,  27,  32,  36,  40,  43,  45,  46, 100]
    detectable = [2,   3,   6,   7,   8,   9,  10,  12,  16, 18,  22,  25,  27,  32,  36,  40,  43,  45, 100] #detectable
    #undetectable = [20, 23, 46] #undetectable in F200W
    undetectable = [2,8,20,23,46,100] #undectable in >2 filters
    
    df=pd.read_pickle('/home/mmarshal/BLUETIDES/BlueTides/PIG_208/processed_data/quasarDatabase_processed.pkl')
    df=df[df['Sample']=='SDSS']
    
    success=np.where(df[filt])[0]
    #print(df[filt])
    print(success)
 
    if 'test' in argv:
        to_plot = to_plot[0:1]

    fig = pp.figure(figsize=(6.8, 8))
    #grid = ImageGrid(fig, 111, nrows_ncols=(4, int(np.ceil(len(to_plot)/4))), axes_pad=0.1,
    #                 share_all=True, label_mode='L',
    #
    #             cbar_location='right', cbar_mode='single')
    grid1 = ImageGrid(fig, (0.05, 0.05, 0.3, 0.9), nrows_ncols=(int(np.ceil(len(to_plot)/3)), 2), axes_pad=[0.1,0.1],
                     share_all=True, label_mode='L',cbar_mode='None')

    grid2 = ImageGrid(fig, (0.3339, 0.164, 0.3, 0.786), nrows_ncols=(int(np.ceil(len(to_plot)/3))-1, 2), axes_pad=[0.1,0.1],
                     share_all=True,label_mode='L',cbar_mode='None')

    grid3 = ImageGrid(fig, (0.63, 0.164, 0.3, 0.786), nrows_ncols=(int(np.ceil(len(to_plot)/3))-1, 2), axes_pad=[0.1,0.1],
                     share_all=True, label_mode='L',
                     cbar_location='right', cbar_mode='single',cbar_pad=0.1,cbar_size=0.1)

    ii=0
    col1=-2
    col2=-2
    col3=-2
    for jj,quasar in enumerate(to_plot):
        if np.mod(jj,3)==0:
          col1+=2
          ii=col1
          grid=grid1
        elif np.mod(jj,3)==1:
          col2+=2
          ii=col2
          grid=grid2
        elif np.mod(jj,3)==2:
          col3+=2
          ii=col3
          grid=grid3
        quasar = 'SDSS_' + str(quasar)
        plot_models(quasar,filt)

        if jj in success:
          mark=r'$\checkmark$'
          mark_col='limegreen'
        else:
          mark=r'$\times$'
          mark_col='red'
        grid[ii].text(0.25,-0.52,mark,color=mark_col,fontsize=20)
        #grid[ii].text(-0.4,0.4,quasar,color=mark_col)#,fontsize=10)
      
    
    #if int(quasar.split('_')[-1]) in undetectable:
    #  mark=r'$\times$'
    #  mark_col='red'
    #else:
   
    xy_format = pp.FormatStrFormatter(r'$%0.1f^{\prime\prime}$')
    for ax in grid1:
        ax.set_xticks(_xytix)
        ax.set_yticks(_xytix)
        ax.xaxis.set_major_formatter(xy_format)
        ax.yaxis.set_major_formatter(xy_format)
    for ax in grid2:
        ax.set_xticks(_xytix)
        ax.set_yticks(_xytix)
        ax.set_yticklabels(['','',''])
        ax.set_xticklabels(_xytix)
        ax.xaxis.set_major_formatter(xy_format)
        #ax.yaxis.set_major_formatter(xy_format)
    for ii,ax in enumerate(grid3):
        ax.set_yticks(_xytix)
        ax.set_yticklabels(['','',''])
        ax.set_xticks(_xytix)
        ax.set_xticklabels(_xytix)
        ax.xaxis.set_major_formatter(xy_format)
        #ax.yaxis.set_major_formatter(xy_format)
    pp.subplots_adjust(left=0.08, bottom=0.1, right=0.91, top=0.92)

    grid1[0].set_title('PSF-Subtracted')#,fontsize=9)
    grid2[0].set_title('PSF-Subtracted')#,fontsize=9)
    grid3[0].set_title('PSF-Subtracted')#,fontsize=9)
    grid1[1].set_title('True Host')#,fontsize=9)
    grid2[1].set_title('True Host')#,fontsize=9)
    grid3[1].set_title('True Host')#,fontsize=9)

    if filt=='F200W':
      pp.savefig('SDSS_z7_trueVsResiduals_appendix.pdf')
    else:
      pp.savefig('SDSS_z7_trueVsResiduals_appendix_{}.pdf'.format(filt))
    #pp.savefig('SDSS_z7_trueVsResiduals_appendix_nonSmooth.pdf')
    pp.show() 
    pp.close(fig)
    
