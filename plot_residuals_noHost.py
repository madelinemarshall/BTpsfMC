from __future__ import division
import numpy as np
import matplotlib.pyplot as pp
import pyregion
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.transforms import Affine2D
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter

from astropy.visualization import AsinhStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from matplotlib import rc
rc('font', family='serif')

_stamp_pat = 'data/sci_mock_{}.fits'
_psfresid_pat = 'runJWST/SDSS_z7_SN_noHost/mcmc_out_mock_{}_point_source_subtracted.fits'
_true_pat = 'data/sci_mock_JWST_{}_{}_onlyHost.fits'
_mag_zp = 25.9463

_stretch = AsinhStretch()
#_stretch.a = (0.01 - 0.0001)/2 / (0.01+0.0001)
#_pnorm = ImageNormalize(vmin=-0.0001, vmax=0.01, stretch=_stretch, clip=True)
_stretch.a = (0.1 - 0.00001)/2 / (0.1+0.00001)
_pnorm = ImageNormalize(vmin=-0.00001, vmax=0.1, stretch=_stretch, clip=True)
_axis_range = [-1.1,1.1,-1.1,1.1]#[-2.5, 2.5, -2.5, 2.5]  # in arcsec
#_xytix = [-3,-2, -1, 0, 1, 2,3]  # in arcsec
_xytix = [-1, 0, 1]  # in arcsec
_coltix = np.array([25,26,27,28])  # in mag/arcsec**2

gray_r = pp.cm.cmap_d['Spectral_r']


def mag_to_flux(mag, zp=0.0, scale=(1.0, 1.0)):
    return 10**(-0.4*(mag - zp)) * np.prod(scale)


def plot_models(quasar, filt, save_name=None):
    psfresid = fits.getdata(_psfresid_pat.format(quasar))
    qdir = 'JWST_F200W_'+quasar.split('_',1)[1]
    stamp = fits.getdata(_stamp_pat.format(qdir))

    #cent=int(len(quasardata)/2)
    #flux_ratio=psfresid[cent,cent]/quasardata[cent,cent]
    #print('Flux ratio: ',psfresid[95,95]/quasar[95,95])
    #if flux_ratio>0.05:
    #  print(flux_ratio,quasar)
   
   
    #psfresid_smooth = gaussian_filter(psfresid, (2, 2))
    resid_smooth = gaussian_filter(psfresid, (2, 2))
    true_smooth = gaussian_filter(stamp, (2, 2))

    center = np.array(psfresid.shape)[::-1]/2
    pxscale = 0.063/2 #arcsec
    extents = np.array([-center[0], center[0],
               -center[1], center[1]])*pxscale

    #plot_panels = [psfresid, 'Point Source\nSubtracted']
    plot_panels = [resid_smooth, 'Point Source\nSubtracted']

    #for ind,data in enumerate([resid_smooth,true_smooth]):
    for ind,data in enumerate([stamp,psfresid]):
      if ind==0:
        grid_ind=ii
        grid[grid_ind].set_title(filt)
      else:
        grid_ind=ii+len(to_plot)
      im = grid[grid_ind].imshow(data, extent=extents, origin='lower',
                               cmap=gray_r, norm=_pnorm,
                               interpolation='nearest')
      grid[grid_ind].axis(_axis_range)
    
      ticks = mag_to_flux(_coltix, zp=_mag_zp, scale=pxscale)
      cbar = pp.colorbar(im, cax=grid.cbar_axes[0], ticks=ticks)
    cbar.set_ticklabels(_coltix)

    #grid[ii].set_title(quasar)
    grid.cbar_axes[0].set_ylabel('mag arcsec$^{-2}$')
    grid.cbar_axes[0].set_xlabel('mag arcsec$^{-2}$')


if __name__ == '__main__':
    from sys import argv
    # import glob
    to_plot = [2] #[2,   3,   6,   7,   8,   9,  10,  12,  16, 18,  20,  22,  23,  25,  27,  32,  36,  40,  43,  45,  46, 100]
    filters = ['F200W']

    if 'test' in argv:
        to_plot = to_plot[0:1]

    fig = pp.figure(figsize=(4.8, 2.5))
    #grid = ImageGrid(fig, 111, nrows_ncols=(2,len(to_plot)), axes_pad=0.1,
    grid = ImageGrid(fig, 111, nrows_ncols=(1,2), axes_pad=0.1,
                     share_all=True, label_mode='L',
                     cbar_location='right', cbar_mode='single')
    ii=0

    for qq in to_plot:
      quasar = 'JWST_SDSS_' + str(qq)
      for filt in filters:
        save_name = 'output_image_{}.pdf'.format(quasar) if 'save' in argv else None
        plot_models(quasar, filt, save_name=save_name)
        ii+=1
   
    xy_format = pp.FormatStrFormatter(r'$%0.0f^{\prime\prime}$')
    for ax in grid:
        ax.set_xticks(_xytix)
        ax.set_yticks(_xytix)
        ax.xaxis.set_major_formatter(xy_format)
        ax.yaxis.set_major_formatter(xy_format)

    grid[0].set_title('Quasar',fontsize=10)
    grid[1].set_title('PSF Subtracted',fontsize=10)
    pp.subplots_adjust(left=0.08, bottom=0.08, right=0.85, top=0.92)
    pp.savefig('residuals_noHost_JWST.pdf'.format(qq))
    pp.show() 
    pp.close(fig)
