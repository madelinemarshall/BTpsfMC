from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib.colors import SymLogNorm
import sys
import pandas as pd

import pyregion
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.transforms import Affine2D
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter

from astropy.visualization import AsinhStretch
from astropy.visualization.mpl_normalize import ImageNormalize

_stamp_pat = '../data/sci_mock_{}.fits'
_model_pat = 'mcmc_out_mock_{}_convolved_model.fits'
_mask_pat = '../data/region.reg'
_psfresid_pat = 'SDSS_z7_{}/mcmc_out_mock_JWST_SDSS_{}_point_source_subtracted.fits'
_rawmodel_pat = 'mcmc_out_mock_{}_raw_model.fits'
_resid_pat = 'mcmc_out_mock_{}_residual.fits'
#_stamp_pat = '../data/sci_mock_{}.fits'
#_model_pat = 'mcmc_out_convolved_model.fits'
#_mask_pat = '../data/region.reg'
#_psfresid_pat = 'mcmc_out_point_source_subtracted.fits'
#_rawmodel_pat = 'mcmc_out_raw_model.fits'
#_resid_pat = 'mcmc_out_residual.fits'
_mag_zp = 25.9463

_stretch = AsinhStretch()
_stretch.a = (0.05 - 0.0005)/2 / (0.05+0.0005)
_pnorm = ImageNormalize(vmin=-0.0005, vmax=0.05, stretch=_stretch, clip=True)
_axis_range = [-2,2,-2,2]#[-2.5, 2.5, -2.5, 2.5]  # in arcsec
#_xytix = [-3,-2, -1, 0, 1, 2,3]  # in arcsec
_xytix = [-1, 0, 1]  # in arcsec
_coltix = np.array([23, 24, 25, 26])  # in mag/arcsec**2

gray_r = plt.cm.cmap_d['nipy_spectral']


def mag_to_flux(mag, zp=0.0, scale=(1.0, 1.0)):
    return 10**(-0.4*(mag - zp)) * np.prod(scale)


def plot_models(filt,ind,kk, save_name=None):
    if (ff=='F200W'):
        psfresid = fits.getdata('SDSS_z7_SN/mcmc_out_mock_JWST_SDSS_{}_point_source_subtracted.fits'.format(ind))
    else:
        psfresid = fits.getdata('SDSS_z7_{}/mcmc_out_mock_JWST_SDSS_{}_point_source_subtracted.fits'.format(filt,ind))
    
    #psfresid_smooth = gaussian_filter(psfresid, (2, 2))
    resid_smooth = gaussian_filter(psfresid, (2, 2))

    center = np.array(psfresid.shape)[::-1]/2
    pxscale = 0.13/2 #arcsec
    extents = np.array([-center[0], center[0],
               -center[1], center[1]])*pxscale

    #plot_panels = [psfresid, 'Point Source\nSubtracted']
    plot_panels = [resid_smooth, 'Point Source\nSubtracted']


    im = grid[kk].imshow(plot_panels[0], extent=extents, origin='lower',
                               cmap=gray_r, norm=_pnorm,
                               interpolation='nearest')
    grid[kk].set_xlabel(plot_panels[1],fontsize=9)
    grid[kk].axis(_axis_range)
    
    ticks = mag_to_flux(_coltix, zp=_mag_zp, scale=pxscale)
    cbar = plt.colorbar(im, cax=grid.cbar_axes[0], ticks=ticks)
    cbar.set_ticklabels(_coltix)
    grid.cbar_axes[0].set_ylabel('mag arcsec$^{-2}$')

    grid[kk].set_title(ff)

def plot_true(filt,ind,kk, save_name=None):
    psfresid = fits.getdata('../data/sci_mock_JWST_{}_SDSS_{}.fits'.format(filt,ind))
    
    #psfresid_smooth = gaussian_filter(psfresid, (2, 2))
    resid_smooth = gaussian_filter(psfresid, (2, 2))

    center = np.array(psfresid.shape)[::-1]/2
    pxscale = 0.13/2 #arcsec
    extents = np.array([-center[0], center[0],
               -center[1], center[1]])*pxscale

    #plot_panels = [psfresid, 'Point Source\nSubtracted']
    plot_panels = [resid_smooth, 'Point Source\nSubtracted']


    im = grid[kk].imshow(plot_panels[0], extent=extents, origin='lower',
                               cmap=gray_r, norm=_pnorm,
                               interpolation='nearest')
    grid[kk].set_xlabel(plot_panels[1],fontsize=9)
    grid[kk].axis(_axis_range)
    
    ticks = mag_to_flux(_coltix, zp=_mag_zp, scale=pxscale)
    cbar = plt.colorbar(im, cax=grid.cbar_axes[0], ticks=ticks)
    cbar.set_ticklabels(_coltix)
    grid.cbar_axes[0].set_ylabel('mag arcsec$^{-2}$')


if __name__=='__main__':

    title='SDSS'
    #indices = [2,   3,   6,   7,   8,   9,  10,  12,  16,  18,  20,  22,  23,  25,  27,  32,  36,  40,   43,  45,  46, 100]
    ii = 3#,   7,   8,   9,  10,  12,  16,  18,  20,  22,  23,  25,  27,  32,  36,  40,   43,  45,  46, 100]
    filters = ['F115W','F150W','F200W','F277W','F356W','F444W']
    
    fig = plt.figure(figsize=(10, 4))
    grid = ImageGrid(fig, 111, nrows_ncols=(2, len(filters)), axes_pad=0.1,
                     share_all=True, label_mode='L',
                     cbar_location='right', cbar_mode='single')

    df=pd.read_pickle('/home/mmarshal/BLUETIDES/BlueTides/PIG_208/processed_data/quasarDatabase.pkl')
    df['Success']=""
    df['Fit_MUV_gal']=""
    df['Fit_MUV_AGN']=""
    
    for kk,ff in enumerate(filters):
      print(ff)
      counts = fits.getdata('../data/sci_mock_JWST_{}_SDSS_{}_host_SN.fits'.format(ff,ii))
      if (ff=='F200W'):
        head = fits.getheader('SDSS_z7_SN/mcmc_out_mock_JWST_SDSS_{}_residual.fits'.format(ii))
      else:
        head = fits.getheader('SDSS_z7_{}/mcmc_out_mock_JWST_SDSS_{}_residual.fits'.format(ff,ii))
      
      gal_fit=float(head['2SER_MAG'].split(' ')[0])
      df.loc[(df['Sample']==title)&(df['Index']==ii),'Fit_MUV_gal']=gal_fit-46.99
      gal_true=float((df.loc[(df['Sample']==title)&(df['Index']==ii)]['MUV_gal_dust'].to_numpy()+46.99)[0])
      print(gal_fit,gal_true)

      print('Subtraction fit - BT: ',gal_fit-gal_true)
      
      quasar_fit=float(head['1PS_MAG'].split(' ')[0])
      df.loc[(df['Sample']==title)&(df['Index']==ii),'Fit_MUV_AGN']=quasar_fit-46.99
      quasar_true=float((df.loc[(df['Sample']==title)&(df['Index']==ii)]['MUV_AGN'].to_numpy()+46.99)[0])

      print('Quasar subtraction fit - BT: ',quasar_fit-quasar_true)

      plot_models(ff,ii,kk)
      plot_true(ff,ii,kk+len(filters))
    
    xy_format = plt.FormatStrFormatter(r'$%0.0f^{\prime\prime}$')
    for ax in grid:
        ax.set_xticks(_xytix)
        ax.set_yticks(_xytix)
        ax.xaxis.set_major_formatter(xy_format)
        ax.yaxis.set_major_formatter(xy_format)
    plt.subplots_adjust(left=0.08, bottom=0.28, right=0.91, top=0.92)
    plt.show() 
    plt.close(fig)
