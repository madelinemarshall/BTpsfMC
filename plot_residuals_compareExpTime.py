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

_stamp_pat = 'data/sci_mock_{}_onlyHost_{}00s.fits'
_psfresid_pat = 'runJWST/SDSS_z7_{}00s/mcmc_out_mock_{}_point_source_subtracted.fits'
_stamp_pat_10 = 'data/sci_mock_{}_onlyHost.fits'
_psfresid_pat_10 = 'runJWST/SDSS_z7_SN/mcmc_out_mock_{}_point_source_subtracted.fits'
_mag_zp = 25.9463

_stretch = AsinhStretch()
_stretch.a = (0.01 - 0.0001)/2 / (0.01+0.0001)
_pnorm = ImageNormalize(vmin=-0.0001, vmax=0.01, stretch=_stretch, clip=True)
_axis_range = [-0.6,0.6,-0.6,0.6]#[-2.5, 2.5, -2.5, 2.5]  # in arcsec
#_xytix = [-3,-2, -1, 0, 1, 2,3]  # in arcsec
_xytix = [-0.5, 0, 0.5]  # in arcsec
_coltix = np.array([27,28,29])  # in mag/arcsec**2

gray_r = pp.cm.cmap_d['Spectral_r']#'nipy_spectral']


def mag_to_flux(mag, zp=0.0, scale=(1.0, 1.0)):
    return 10**(-0.4*(mag - zp)) * np.prod(scale)


def plot_models(quasar, ii, save_name=None):
    jj=ii*4
    for expTime in [10,25,50]:
      psfresid = fits.getdata(_psfresid_pat.format(expTime,quasar))
      psfresid_smooth = gaussian_filter(psfresid, (1, 1))
      qdir = 'JWST_F200W_'+quasar.split('_',1)[1]
      #stamp = fits.getdata(_stamp_pat.format(qdir,expTime))
      #stamp_smooth = gaussian_filter(stamp, (2, 2))

      center = np.array(psfresid.shape)[::-1]/2
      pxscale = 0.031/2 #arcsec
      extents = np.array([-center[0], center[0],
               -center[1], center[1]])*pxscale

      #plot_panels = [psfresid, 'Point Source\nSubtracted']
      plot_panels = [psfresid_smooth, 'Point Source\nSubtracted']

      im = grid[jj].imshow(plot_panels[0], extent=extents, origin='lower',
                               cmap=gray_r, norm=_pnorm,
                               interpolation='nearest')
      jj+=1

    ###exp time = 10ks
    psfresid = fits.getdata(_psfresid_pat_10.format(quasar))
    psfresid_smooth = gaussian_filter(psfresid, (1, 1))
    qdir = 'JWST_F200W_'+quasar.split('_',1)[1]
    #stamp = fits.getdata(_stamp_pat_10.format(qdir))
    #stamp_smooth = gaussian_filter(stamp, (2, 2))

    #plot_panels = [psfresid, 'Point Source\nSubtracted']
    plot_panels = [psfresid_smooth, 'Point Source\nSubtracted']

    im = grid[jj].imshow(plot_panels[0], extent=extents, origin='lower',
                               cmap=gray_r, norm=_pnorm,
                               interpolation='nearest')
    jj+=1
    
    ###True Host
    #psfresid = fits.getdata(_psfresid_pat_10.format(quasar))
    #psfresid_smooth = gaussian_filter(psfresid, (2, 2))
    qdir = 'JWST_F200W_'+quasar.split('_',1)[1]
    stamp = fits.getdata(_stamp_pat_10.format(qdir))
    stamp_smooth = gaussian_filter(stamp, (1, 1))

    #plot_panels = [stamp]
    plot_panels = [stamp_smooth]

    im = grid[jj].imshow(plot_panels[0], extent=extents, origin='lower',
                               cmap=gray_r, norm=_pnorm,
                               interpolation='nearest')
    grid[jj].axis(_axis_range)
    
    ticks = mag_to_flux(_coltix, zp=_mag_zp, scale=pxscale)
    cbar = pp.colorbar(im, cax=grid.cbar_axes[0], ticks=ticks)
    cbar.set_ticklabels(_coltix)
    grid.cbar_axes[0].set_ylabel('mag arcsec$^{-2}$')
    grid.cbar_axes[0].set_xlabel('mag arcsec$^{-2}$')
    
    
    mark=r'$\times$'
    mark_col='red'
    grid[0].text(0.35,-0.55,mark,color=mark_col,fontsize=22)
    mark=r'$\checkmark$'
    mark_col='limegreen'
    for ii in range(1,4):
      grid[ii].text(0.35,-0.55,mark,color=mark_col,fontsize=22)
 

    grid[0].set_title('1 ks',fontsize=10)
    grid[1].set_title('2.5 ks',fontsize=10)
    grid[2].set_title('5 ks',fontsize=10)
    grid[3].set_title('10 ks',fontsize=10)
    grid[4].set_title('True Image (10 ks)',fontsize=10)
    #grid[ii].set_title(quasar)

if __name__ == '__main__':
    from sys import argv
    # import glob
    to_plot = [12]#,8]

    if 'test' in argv:
        to_plot = to_plot[0:1]

    fig = pp.figure(figsize=(10, 2.6))
    grid = ImageGrid(fig, 111, nrows_ncols=(len(to_plot), 5), axes_pad=0.1,
                     share_all=True, label_mode='L',
                     cbar_location='right', cbar_mode='single')
   
    ii=0 
    for quasar in to_plot:
        quasar = 'JWST_SDSS_' + str(quasar)
        save_name = 'output_image_{}.pdf'.format(quasar) if 'save' in argv else None
        plot_models(quasar, ii, save_name=save_name)
        ii+=1
   
    xy_format = pp.FormatStrFormatter(r'$%0.1f^{\prime\prime}$')
    for ax in grid:
        ax.set_xticks(_xytix)
        ax.set_yticks(_xytix)
        ax.xaxis.set_major_formatter(xy_format)
        ax.yaxis.set_major_formatter(xy_format)
    pp.subplots_adjust(left=0.06, bottom=0.06, right=0.9, top=0.94)
    pp.savefig('residual_compare_expTime.pdf')
    pp.show() 
    pp.close(fig)
