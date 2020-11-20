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
_axis_range = [-2,2,-2,2]#[-2.5, 2.5, -2.5, 2.5]  # in arcsec
#_xytix = [-3,-2, -1, 0, 1, 2,3]  # in arcsec
_xytix = [-1, 0, 1]  # in arcsec
_coltix = np.array([23, 24, 25, 26])  # in mag/arcsec**2

gray_r = pp.cm.cmap_d['Spectral_r']


def mag_to_flux(mag, zp=0.0, scale=(1.0, 1.0)):
    return 10**(-0.4*(mag - zp)) * np.prod(scale)


def plot_models(quasar,filt):
    psfresid = fits.getdata(_psfresid_pat.format(filt,quasar))
    #psfresid_smooth = gaussian_filter(psfresid, (2, 2))
    resid_smooth = gaussian_filter(psfresid, (2, 2))

    center = np.array(psfresid.shape)[::-1]/2
    pxscale = 0.13/2 #arcsec
    extents = np.array([-center[0], center[0],
               -center[1], center[1]])*pxscale

    #plot_panels = [psfresid, 'Point Source\nSubtracted']
    plot_panels = [resid_smooth, 'Point Source\nSubtracted']


    im = grid[ii].imshow(plot_panels[0], extent=extents, origin='lower',
                               cmap=gray_r, norm=_pnorm,
                               interpolation='nearest')
    grid[ii].axis(_axis_range)
    
    ticks = mag_to_flux(_coltix, zp=_mag_zp, scale=pxscale)
    cbar = pp.colorbar(im, cax=grid.cbar_axes[0])#, ticks=ticks)
    #cbar.set_ticklabels(_coltix)
    grid.cbar_axes[0].set_ylabel('mag arcsec$^{-2}$')
    grid.cbar_axes[0].set_xlabel('mag arcsec$^{-2}$')

    #grid[ii].set_title(quasar)

if __name__ == '__main__':
    from sys import argv
    # import glob
    #to_plot=[4,5,11,13,14,15,19,26,28,31,34,35,37,38,42,44,49,51,52,53,54,56,57,61,62,63,64,67,68,71,77,78,82,85,86,87,94,97,98,101,103,105,106,110,111,112,113,114,119,124,125,126,128,129,130,133,138,143,145,147,148,149,151,156,158,159,161,163,165,172,173,177,178,179,181,182,184,186,189,191,195,196,199,200,202,204,205,206,207,214,215,218,219,221,222,223,226,231,233,234,235,241,244,246,248,249,251,252,255,258,259,261,265,266,268,269,270,272,274,275,278,281,290,291,292,297,298,302,304,306,308,311,315,317,319,324,328,335,337,345,346,348,353,363,366,367,369,380,384,388,390,396,409,410,427,429,437,442,459,460,461,464,465,477,483,487,505,519,528,546,553,561,572,606,612] #detectable
    #to_plot=[52,53,56,82,86,87,177,178,181,182,184,189,214,218,219,231,233,234,235,248,249,251,427,437] #detectable around undetectable
    to_plot=[ 54,85,179,186,215,234,251,429] #undetectable

    if 'test' in argv:
        to_plot = to_plot[0:1]

    fig = pp.figure(figsize=(10, 10))
    grid = ImageGrid(fig, 111, nrows_ncols=(4, int(np.ceil(len(to_plot)/4))), axes_pad=0.1,
                     share_all=True, label_mode='L',
                     cbar_location='right', cbar_mode='single')
   
    if len(argv)>1:
      filt=str(argv[1])
    else:
      filt='F200W'

    ii=0 
    for quasar in to_plot:
        quasar = 'CO_' + str(quasar)
        plot_models(quasar,filt)
        ii+=1
   
    xy_format = pp.FormatStrFormatter(r'$%0.0f^{\prime\prime}$')
    for ax in grid:
        ax.set_xticks(_xytix)
        ax.set_yticks(_xytix)
        ax.xaxis.set_major_formatter(xy_format)
        ax.yaxis.set_major_formatter(xy_format)
    pp.subplots_adjust(left=0.08, bottom=0.28, right=0.91, top=0.92)

    pp.savefig('SDSS_z7_trueHosts.pdf')
    pp.show() 
    pp.close(fig)
    
