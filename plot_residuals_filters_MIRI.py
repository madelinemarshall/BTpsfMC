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

_psfresid_pat = 'runJWST/SDSS_z7_{}/mcmc_out_mock_{}_point_source_subtracted.fits'
_psfresid_pat_F200W = 'runJWST/SDSS_z7_SN/mcmc_out_mock_{}_point_source_subtracted.fits'
_true_pat = 'data/sci_mock_JWST_{}_{}_onlyHost.fits'
_mag_zp = 25.9463

_stretch = AsinhStretch()
#_stretch.a = (0.01 - 0.0001)/2 / (0.01+0.0001)
#_pnorm = ImageNormalize(vmin=-0.0001, vmax=0.01, stretch=_stretch, clip=True)
#_xytix = [-3,-2, -1, 0, 1, 2,3]  # in arcsec
_xytix = [-0.5, 0, 0.5]  # in arcsec

gray_r = pp.cm.cmap_d['Spectral_r']


def mag_to_flux(mag, zp=0.0, scale=(1.0, 1.0)):
    return 10**(-0.4*(mag - zp)) * np.prod(scale)


def plot_models(quasar, filt, ii, save_name=None):
    q_ind = quasar.split('_',1)[1]#+'_host'
    if filt=='F200W':
      psfresid = fits.getdata(_psfresid_pat_F200W.format(quasar))
    else:
      psfresid = fits.getdata(_psfresid_pat.format(filt,quasar))
    trueHost = fits.getdata(_true_pat.format(filt,q_ind))
    #quasardata = fits.getdata(_quasar_pat.format(qdir))

    #cent=int(len(quasardata)/2)
    #flux_ratio=psfresid[cent,cent]/quasardata[cent,cent]
    #print('Flux ratio: ',psfresid[95,95]/quasar[95,95])
    #if flux_ratio>0.05:
    #  print(flux_ratio,quasar)
   
   
    #psfresid_smooth = gaussian_filter(psfresid, (2, 2))
    resid_smooth = gaussian_filter(psfresid, (1, 1))
    true_smooth = gaussian_filter(trueHost, (1, 1))

    center = np.array(psfresid.shape)[::-1]/2
    if filt in ['F277W','F356W','F444W']:
      pxscale = 0.063/2 #arcsec
      nfilt = 3
      #_stretch.a = (0.03 - 0.00001)/2 / (0.03+0.00001)
      #_pnorm = ImageNormalize(vmin=-0.00001*2.4, vmax=0.03*2.4, stretch=_stretch, clip=True)
      _axis_range = [-0.6,0.6,-0.6,0.6]#[-2.5, 2.5, -2.5, 2.5]  # in arcsec
      _coltix = np.array([26,27,28])  # in mag/arcsec**2
      _stretch.a = (0.06 - 0.00005)/2 / (0.06+0.00005)
      _pnorm = ImageNormalize(vmin=-0.00005, vmax=0.06, stretch=_stretch, clip=True)
    elif filt in ['F560W','F770W']:
      pxscale = 0.11/2
      nfilt = 2
      #_axis_range = [-0.8,0.8,-0.8,0.8]
      _axis_range = [-0.6,0.6,-0.6,0.6]#[-2.5, 2.5, -2.5, 2.5]  # in arcsec
      _coltix = np.array([26,27,28])  # in mag/arcsec**2
      _stretch.a = (0.1 - 0.00005)/2 / (0.1+0.00005)
      _pnorm = ImageNormalize(vmin=-0.00005, vmax=0.1, stretch=_stretch, clip=True)
    else:
      pxscale = 0.031/2
      nfilt = 3
      _axis_range = [-0.6,0.6,-0.6,0.6]#[-2.5, 2.5, -2.5, 2.5]  # in arcsec
      _coltix = np.array([27,28,29])  # in mag/arcsec**2
      _stretch.a = (0.01 - 0.00005)/2 / (0.01+0.00005)
      _pnorm = ImageNormalize(vmin=-0.00005, vmax=0.01, stretch=_stretch, clip=True)
    extents = np.array([-center[0], center[0],
               -center[1], center[1]])*pxscale

    for ind,data in enumerate([resid_smooth,true_smooth]):
    #for ind,data in enumerate([psfresid,trueHost]):
      if ind==0:
        grid_ind=ii
        grid[grid_ind].set_title(filt)
      else:
        grid_ind=ii+nfilt
      #print(filt,quasar,ind,grid_ind)
      im = grid[grid_ind].imshow(data, extent=extents, origin='lower',
                               cmap=gray_r, norm=_pnorm,
                               interpolation='nearest')
      grid[grid_ind].axis(_axis_range)
    
      ticks = mag_to_flux(_coltix, zp=_mag_zp, scale=pxscale)
      cbar = pp.colorbar(im, cax=grid.cbar_axes[grid_ind], ticks=ticks)
      cbar.set_ticklabels(_coltix)
      grid.cbar_axes[grid_ind].set_ylabel('mag arcsec$^{-2}$')
      grid.cbar_axes[grid_ind].set_xlabel('mag arcsec$^{-2}$')
      grid.cbar_axes[grid_ind].yaxis.set_label_coords(3.8,0.5)

    #grid[ii].set_title(quasar)

if __name__ == '__main__':
    from sys import argv
    # import glob
    to_plot = [2] #3[2,   3,   6,   7,   8,   9,  10,  12,  16, 18,  20,  22,  23,  25,  27,  32,  36,  40,  43,  45,  46, 100]
    filters = ['F115W','F150W','F200W','F277W','F356W','F444W','F560W','F770W']

    if 'test' in argv:
        to_plot = to_plot[0:1]

    for qq in to_plot:
      fig = pp.figure(figsize=(9.8, 6))
      #grid = ImageGrid(fig, 111, nrows_ncols=(2,len(filters)), axes_pad=0.1,
      #               share_all=True, label_mode='L',
      #               cbar_location='right', cbar_mode='each')
      grid1 = ImageGrid(fig, (0.08, 0.54, 0.4, 0.42), nrows_ncols=(2,3), axes_pad=[0.1,0.1],
                     share_all=True, label_mode='L',
                     cbar_location='right', cbar_mode='single',cbar_pad=0.1,cbar_size=0.1)

      grid2 = ImageGrid(fig, (0.54, 0.54, 0.4, 0.42), nrows_ncols=(2,3), axes_pad=[0.1,0.1],
                     share_all=True,label_mode='L',
                     cbar_location='right', cbar_mode='single',cbar_pad=0.1,cbar_size=0.1)

      #grid3 = ImageGrid(fig, (0.0761, 0.05, 0.28, 0.41), nrows_ncols=(2,2), axes_pad=[0.1,0.1],
      #               share_all=True, label_mode='L',
      #               cbar_location='right', cbar_mode='single',cbar_pad=0.1,cbar_size=0.1)
      grid3 = ImageGrid(fig, (0.37, 0.05, 0.28, 0.41), nrows_ncols=(2,2), axes_pad=[0.1,0.1],
                     share_all=True, label_mode='L',
                     cbar_location='right', cbar_mode='single',cbar_pad=0.1,cbar_size=0.1)
 
   
      ii=0 
      col1=-1
      col2=-1
      col3=-1
      quasar = 'JWST_SDSS_' + str(qq)
      for filt in filters:
          if filt in ['F115W','F150W','F200W']:
              col1+=1
              ii=col1
              grid=grid1
          elif filt in ['F277W','F356W','F444W']:
              col2+=1
              ii=col2
              grid=grid2
          else:
              col3+=1
              ii=col3
              grid=grid3
          save_name = 'output_image_{}.pdf'.format(quasar) if 'save' in argv else None
          plot_models(quasar, filt, ii, save_name=save_name)
          ii+=1
   
      xy_format = pp.FormatStrFormatter(r'$%0.1f^{\prime\prime}$')
      for ax in grid1:
        ax.set_xticks(_xytix)
        ax.set_yticks(_xytix)
        ax.xaxis.set_major_formatter(xy_format)
        ax.yaxis.set_major_formatter(xy_format)
      for ax in grid2:
        ax.set_xticks(_xytix)
        ax.set_yticks(_xytix)
        ax.xaxis.set_major_formatter(xy_format)
        #ax.yaxis.set_major_formatter(xy_format)
        ax.set_yticklabels(['','',''])
      for ax in grid3:
        ax.set_xticks(_xytix)
        ax.set_yticks(_xytix)
        ax.xaxis.set_major_formatter(xy_format)
        ax.yaxis.set_major_formatter(xy_format)

      grid1[0].set_ylabel('PSF Subtracted')
      grid1[3].set_ylabel('True Host')
      grid1[0].yaxis.set_label_coords(-0.5,0.5)
      grid1[3].yaxis.set_label_coords(-0.5,0.5)
      grid3[0].set_ylabel('PSF Subtracted')
      grid3[2].set_ylabel('True Host')
      grid3[0].yaxis.set_label_coords(-0.5,0.5)
      grid3[2].yaxis.set_label_coords(-0.5,0.5)
    

      mark=r'$\times$'
      mark_col='red'
      grid3[1].text(0.35,-0.55,mark,color=mark_col,fontsize=22)
    
      mark=r'$\checkmark$'
      mark_col='limegreen'
      for ii in range(0,3):
        grid1[ii].text(0.35,-0.55,mark,color=mark_col,fontsize=22)
        grid2[ii].text(0.35,-0.55,mark,color=mark_col,fontsize=22)
      grid3[0].text(0.35,-0.55,mark,color=mark_col,fontsize=22)
      #pp.subplots_adjust(left=0.08, bottom=0.1, right=0.92, top=0.95)
      pp.savefig('residuals_filter_comparison_MIRI_SDSS_{}.pdf'.format(qq))
      pp.show() 
      pp.close(fig)
