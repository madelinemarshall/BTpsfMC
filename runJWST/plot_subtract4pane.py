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

#_stamp_pat = '../data/sci_CFHQS-J0033-0125_H.fits.gz'#sci_mock_{}.fits'

_stamp_pat = '../data/sci_mock_{}.fits'
_model_pat = 'mcmc_out_mock_{}_convolved_model.fits'
_mask_pat = '../data/region.reg'
_psfresid_pat = 'mcmc_out_mock_{}_point_source_subtracted.fits'
_rawmodel_pat = 'mcmc_out_mock_{}_raw_model.fits'
_resid_pat = 'mcmc_out_mock_{}_residual.fits'
#_stamp_pat = '../data/sci_mock_{}.fits'
#_model_pat = 'mcmc_out_convolved_model.fits'
#_mask_pat = '../data/region.reg'
#_psfresid_pat = 'mcmc_out_point_source_subtracted.fits'
#_rawmodel_pat = 'mcmc_out_raw_model.fits'
#_resid_pat = 'mcmc_out_residual.fits'
_mag_zp = {'F125W': 26.2303, 'F160W': 25.9463}

_stretch = AsinhStretch()
_stretch.a = (0.05 - 0.0005)/2 / (0.05+0.0005)
_pnorm = ImageNormalize(vmin=-0.0005, vmax=0.05, stretch=_stretch, clip=True)
_axis_range = [-2,2,-2,2]#[-2.5, 2.5, -2.5, 2.5]  # in arcsec
#_xytix = [-3,-2, -1, 0, 1, 2,3]  # in arcsec
_xytix = [-1, 0, 1]  # in arcsec
_coltix = np.array([23, 24, 25, 26])  # in mag/arcsec**2

gray_r = pp.cm.cmap_d['nipy_spectral']


def mag_to_flux(mag, zp=0.0, scale=(1.0, 1.0)):
    return 10**(-0.4*(mag - zp)) * np.prod(scale)


def plot_models(quasar, save_name=None):
    qdir = 'JWST_F200W_'+quasar.split('_',1)[1]+'_onlyHost'
    stamp = fits.getdata(_stamp_pat.format(qdir))
    #stamp = fits.getdata(_stamp_pat)
    wcs = WCS(fits.getheader(_stamp_pat.format(qdir)))
    model = fits.getdata(_model_pat.format(quasar))
    psfresid = fits.getdata(_psfresid_pat.format(quasar))
    rawmodel = fits.getdata(_rawmodel_pat.format(quasar))
    filt = 'F160W'
    residual = fits.getdata(_resid_pat.format(quasar))

    #psfresid_smooth = gaussian_filter(psfresid, (2, 2))
    resid_smooth = gaussian_filter(residual, (2, 2))

    center = np.array(stamp.shape)[::-1]/2
    pxscale = 0.13/2 #arcsec
    extents = np.array([-center[0], center[0],
               -center[1], center[1]])*pxscale

    plot_panels = [
        (stamp, 'Quasar'),
        #(rawmodel, 'Posterior Model'),
        (model, 'Convolved Model'),
        (psfresid, 'Point Source\nSubtracted'),
        (residual, 'Model Subtracted'),
    ]

    regs = pyregion.open(_mask_pat.format(qdir, quasar))

    fig = pp.figure(figsize=(7.5, 2.5))

    grid = ImageGrid(fig, 111, nrows_ncols=(1, len(plot_panels)), axes_pad=0.1,
                     share_all=True, label_mode='L',
                     cbar_location='right', cbar_mode='single')
    for cell, (image, caption) in enumerate(plot_panels):
        im = grid[cell].imshow(image, extent=extents, origin='lower',
                               cmap=gray_r, norm=_pnorm,
                               interpolation='nearest')
        #print(cell,image,caption)
        grid[cell].set_xlabel(caption,fontsize=9)
        #patch_list, artist_list = regs.get_mpl_patches_texts()
    grid[0].axis(_axis_range)
    
    ticks = mag_to_flux(_coltix, zp=_mag_zp[filt], scale=pxscale)
    cbar = pp.colorbar(im, cax=grid.cbar_axes[0], ticks=ticks)
    cbar.set_ticklabels(_coltix)
    grid.cbar_axes[0].set_ylabel('mag arcsec$^{-2}$')

    xy_format = pp.FormatStrFormatter(r'$%0.0f^{\prime\prime}$')
    for ax in grid:
        ax.set_xticks(_xytix)
        ax.set_yticks(_xytix)
        ax.xaxis.set_major_formatter(xy_format)
        ax.yaxis.set_major_formatter(xy_format)

    grid[0].set_ylabel(filt)

    pp.figtext(0.5, 1.0, quasar, va='top', ha='center')
    pp.subplots_adjust(left=0.08, bottom=0.28, right=0.91, top=0.92)
    if save_name is not None:
        pp.savefig(save_name)
    else:
        pp.show()
    pp.close(fig)

if __name__ == '__main__':
    from sys import argv
    # import glob
    to_plot = ['JWST_MMBH','JWST_SDSS','JWST_CO','JWST_WFIRST']
    #to_plot = ['JWST_PSFx15','JWST_PSFx10','JWST_PSFx5','JWST_PSFx1']
    #to_plot = ['JWST_F090W','JWST_F115W','JWST_F150W','JWST_F277W','JWST_F356W','JWST_F444W']
    #to_plot = ['JWST_F200W_0p01','JWST_F200W_0p05']

    if 'test' in argv:
        to_plot = to_plot[0:1]

    for quasar in to_plot:
        save_name = 'output_image_{}.pdf'.format(quasar) if 'save' in argv else None
        plot_models(quasar, save_name=save_name)
