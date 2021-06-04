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

from astropy.visualization import AsinhStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from matplotlib import rc
rc('font', family='serif')
matplotlib.rcParams['font.size'] = (8)

_stamp_pat = 'data/sci_mock_{}_onlyHost.fits'
_psfresid_pat_sm = 'runJWST/SDSS_z7_smooth/mcmc_out_mock_{}_point_source_subtracted.fits'
_psfresid_pat_SN = 'runJWST/SDSS_z7_SN/mcmc_out_mock_{}_point_source_subtracted.fits'
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


def plot_models(quasar, save_name=None):
    psfresid_sm = fits.getdata(_psfresid_pat_sm.format(quasar))
    psfresid_SN = fits.getdata(_psfresid_pat_SN.format(quasar))
    #psfresid_smooth = gaussian_filter(psfresid, (2, 2))
    resid_smooth_SN = gaussian_filter(psfresid_SN, (1, 1))
    resid_smooth_sm = gaussian_filter(psfresid_sm, (1, 1))
    qdir = 'JWST_F200W_'+quasar.split('_',1)[1]
    stamp = fits.getdata(_stamp_pat.format(qdir))
    stamp_smooth = gaussian_filter(stamp, (1, 1))

    center = np.array(psfresid_SN.shape)[::-1]/2
    pxscale = 0.031/2 #arcsec
    extents = np.array([-center[0], center[0],
               -center[1], center[1]])*pxscale

    #plot_panels = [psfresid, 'Point Source\nSubtracted']
    plot_panels = [stamp_smooth,resid_smooth_sm,resid_smooth_SN]

    for jj,dat in enumerate(plot_panels):
      im = grid[jj].imshow(dat, extent=extents, origin='lower',
                               cmap=gray_r, norm=_pnorm,
                               interpolation='nearest')
    grid[jj].axis(_axis_range)
    
    ticks = mag_to_flux(_coltix, zp=_mag_zp, scale=pxscale)
    cbar = pp.colorbar(im, cax=grid.cbar_axes[0], ticks=ticks)
    cbar.set_ticklabels(_coltix)
    grid.cbar_axes[0].set_ylabel('mag arcsec$^{-2}$')
    grid.cbar_axes[0].set_xlabel('mag arcsec$^{-2}$')

    grid[0].set_title('True Host')#,fontsize=10)
    grid[1].set_title('Smooth\nPSF Subtraction')#,fontsize=10)
    grid[2].set_title('Star\nPSF Subtraction')#,fontsize=10)
    #grid[ii].set_title(quasar)

if __name__ == '__main__':
    from sys import argv
    # import glob
    to_plot = [3]

    if 'test' in argv:
        to_plot = to_plot[0:1]

    fig = pp.figure(figsize=(4.5, 1.8))
    grid = ImageGrid(fig, 111, nrows_ncols=(1, 3), axes_pad=0.1,
                     share_all=True, label_mode='L',
                     cbar_location='right', cbar_mode='single')
   
    ii=0 
    for quasar in to_plot:
        quasar = 'JWST_SDSS_' + str(quasar)
        save_name = 'output_image_{}.pdf'.format(quasar) if 'save' in argv else None
        plot_models(quasar, save_name=save_name)
        ii+=1
   
    xy_format = pp.FormatStrFormatter(r'$%0.1f^{\prime\prime}$')
    for ax in grid:
        ax.set_xticks(_xytix)
        ax.set_yticks(_xytix)
        ax.xaxis.set_major_formatter(xy_format)
        ax.yaxis.set_major_formatter(xy_format)
    pp.subplots_adjust(left=0.08, bottom=0.08, right=0.88, top=0.92)
    pp.savefig('residual_compare_PSFs.pdf')
    pp.show() 
    pp.close(fig)
