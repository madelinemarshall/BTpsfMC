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

_psfresid_pat = 'data/sci_mock_JWST_{}_{}_onlyHost.fits'
_mag_zp = 25.9463

_stretch = AsinhStretch()
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
    psfresid = fits.getdata(_psfresid_pat.format(filt,quasar))
    #psfresid_smooth = gaussian_filter(psfresid, (2, 2))
    resid_smooth = gaussian_filter(psfresid, (1, 1))

    center = np.array(psfresid.shape)[::-1]/2
    pxscale = 0.031/2
    extents = np.array([-center[0], center[0],
               -center[1], center[1]])*pxscale

    #plot_panels = [psfresid, 'Point Source\nSubtracted']
    plot_panels = [resid_smooth, 'Point Source\nSubtracted']


    im = grid[ii].imshow(plot_panels[0], extent=extents, origin='lower',
                               cmap=gray_r, norm=_pnorm,
                               interpolation='nearest')
    if int(quasar.split('_')[-1]) in undetectable:
      mark=r'$\times$'
      mark_col='red'
    else:
      mark=r'$\checkmark$'
      mark_col='limegreen'
    grid[ii].axis(_axis_range)
    grid[ii].text(0.3,-0.45,mark,color=mark_col,fontsize=20)
    
    ticks = mag_to_flux(_coltix, zp=_mag_zp, scale=pxscale)
    cbar = pp.colorbar(im, cax=grid.cbar_axes[0], ticks=ticks)
    cbar.set_ticklabels(_coltix)
    grid.cbar_axes[0].set_ylabel('mag arcsec$^{-2}$')
    grid.cbar_axes[0].set_xlabel('mag arcsec$^{-2}$')

    #grid[ii].set_title(quasar)

if __name__ == '__main__':
    from sys import argv
    # import glob
    to_plot = [2,   3,   6,   7,   8,   9,  10,  12,  16, 18,  20,  22,  23,  25,  27,  32,  36,  40,  43,  45,  46, 100]
    detectable = [2,   3,   6,   7,   8,   9,  10,  12,  16, 18,  22,  25,  27,  32,  36,  40,  43,  45, 100] #detectable
    #undetectable = [20, 23, 46] #undetectable in F200W
    undetectable = [2,8,20,23,46,100] #undectable in >2 filters
    

    if 'test' in argv:
        to_plot = to_plot[0:1]

    fig = pp.figure(figsize=(10, 6))
    grid = ImageGrid(fig, 111, nrows_ncols=(4, int(np.ceil(len(to_plot)/4))), axes_pad=0.1,
                     share_all=True, label_mode='L',
                     cbar_location='right', cbar_mode='single')
   
    if len(argv)>1:
      filt=str(argv[1])
    else:
      filt='F200W'

    ii=0 
    for quasar in to_plot:
        quasar = 'SDSS_' + str(quasar)
        plot_models(quasar,filt)
        ii+=1
   
    xy_format = pp.FormatStrFormatter(r'$%0.1f^{\prime\prime}$')
    for ax in grid:
        ax.set_xticks(_xytix)
        ax.set_yticks(_xytix)
        ax.xaxis.set_major_formatter(xy_format)
        ax.yaxis.set_major_formatter(xy_format)
    pp.subplots_adjust(left=0.08, bottom=0.1, right=0.91, top=0.92)
    grid[-1].axis('off')
    grid[-2].axis('off')
    pp.savefig('SDSS_z7_trueHosts.pdf')
    pp.show() 
    pp.close(fig)
    
