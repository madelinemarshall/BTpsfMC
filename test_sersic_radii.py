###Test to see if extending the prior ranges for reff in the Sersic profile PSFMC fits from 0.5-10 to 5-20 pixels performs better (as many clustered near 10). It does not, actually makes quasar fit worse.
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from photutils import CircularAperture,CircularAnnulus
from photutils import aperture_photometry, background
import matplotlib
from matplotlib.colors import SymLogNorm
import sys
import pandas as pd
from scipy.ndimage import gaussian_filter
from astropy.visualization import AsinhStretch, LogStretch, SinhStretch, SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.convolution import Gaussian2DKernel, Tophat2DKernel, Box2DKernel, interpolate_replace_nans
from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid
from scipy.interpolate import interp1d


matplotlib.rcParams['font.size'] = (9)
#plt.rc('text', usetex=True)
plt.rc('font', family='serif')
markers= ['s','^','o','d','h','p']

_stamp_pat = 'data/sci_mock_{}_onlyHost.fits'
_psfresid_pat = 'runJWST/SDSS_z7_SN/mcmc_out_mock_{}_point_source_subtracted.fits'
_psfresid_pat_large = 'runJWST/SDSS_z7_largerSizes/mcmc_out_mock_{}_point_source_subtracted.fits'
_mag_zp = 25.9463

_stretch = AsinhStretch()
_stretch.a = (0.01 - 0.0001)/2 / (0.01+0.0001)
_pnorm = ImageNormalize(vmin=-0.0001, vmax=0.01, stretch=_stretch, clip=True)
_axis_range = [-0.6,0.6,-0.6,0.6]#[-2.5, 2.5, -2.5, 2.5]  # in arcsec
#_xytix = [-3,-2, -1, 0, 1, 2,3]  # in arcsec
_xytix = [-0.5, 0, 0.5]  # in arcsec
_coltix = np.array([27,28,29])  # in mag/arcsec**2

gray_r = plt.cm.cmap_d['Spectral_r']#'nipy_spectral']


def mag_to_flux(mag, zp=0.0, scale=(1.0, 1.0)):
    return 10**(-0.4*(mag - zp)) * np.prod(scale)


def plot_models(quasar, ii, save_name=None):
    jj=ii*2
    ###exp time = 10ks
    psfresid = fits.getdata(_psfresid_pat.format(quasar))
    psfresid_large = fits.getdata(_psfresid_pat_large.format(quasar))
    psfresid_smooth = gaussian_filter(psfresid, (1, 1))
    psfresid_smooth_large = gaussian_filter(psfresid_large, (1, 1))
    qdir = 'JWST_F200W_'+quasar.split('_',1)[1]
    center = np.array(psfresid.shape)[::-1]/2
    pxscale = 0.031/2 #arcsec
    extents = np.array([-center[0], center[0],
               -center[1], center[1]])*pxscale
    #stamp = fits.getdata(_stamp_pat_10.format(qdir))
    #stamp_smooth = gaussian_filter(stamp, (2, 2))

    #plot_panels = [psfresid, 'Point Source\nSubtracted']
    plot_panels = [psfresid_smooth,psfresid_smooth_large]

    im = grid[jj].imshow(plot_panels[0], extent=extents, origin='lower',
			       cmap=gray_r, norm=_pnorm,
			       interpolation='nearest')
    jj+=1
    im = grid[jj].imshow(plot_panels[1], extent=extents, origin='lower',
			       cmap=gray_r, norm=_pnorm,
			       interpolation='nearest')
    jj+=1

    
    ticks = mag_to_flux(_coltix, zp=_mag_zp, scale=pxscale)
    cbar = plt.colorbar(im, cax=grid.cbar_axes[0], ticks=ticks)
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
 
    #grid[ii].set_title(quasar)

    return
                
def sersic_rad(filt,ii):
    if filt=='F200W':
      _pattern = 'runJWST/SDSS_z7_SN/mcmc_out_mock_JWST_SDSS_{}_residual.fits'
      head = fits.getheader(_pattern.format(ii))
    else:
      _pattern = 'runJWST/SDSS_z7_{}/mcmc_out_mock_JWST_SDSS_{}_residual.fits'
      head = fits.getheader(_pattern.format(filt,ii))
    fit_rad=head['2SER_RE'].split(' ')[0]
    err_rad=head['2SER_RE'].split(' ')[-1]
    #fit_sers[jj]=head['2SER_N'].split(' ')[0]
    #err_sers[jj]=head['2SER_N'].split(' ')[-1]
    return fit_rad,err_rad

def sersic_rad_true(filt,ii):
    #test output from larger sizes
    if filt=='F200W':
      _pattern_OH= 'runJWST/SDSS_z7_largerSizes/mcmc_out_mock_JWST_SDSS_{}_residual.fits'
    head_OH = fits.getheader(_pattern_OH.format(ii))
    OH_rad=head_OH['2SER_RE'].split(' ')[0]
    OH_rad_err=head_OH['2SER_RE'].split(' ')[-1]
    return OH_rad,OH_rad_err
      

def ext_dict(dd):
    return np.concatenate(list(dd.values()))


def find_eff_half_light_radius(pix_area,z):
    #Find the area (non-contiguous) which contains 50% of the luminosity
    ordered_l = np.sort(z,axis=None)[::-1]
    total_l=np.sum(ordered_l)
    sum_l = np.cumsum(ordered_l)
    num_above_half=(np.size(ordered_l[sum_l<total_l/2]))+1 #number of pixels containining 50% of luminosity
    #lum_half=ordered_l[sum_l<total_l/2][-1]   

    area = pix_area * num_above_half
    radius = np.sqrt(area/np.pi)
    return radius #, (lum_half,sum_l[sum_l<total_l/2][-1],total_l/2)


def find_half_light_radius(x,y,z):
    centre = np.unravel_index(np.argmax(z), np.array(z).shape) #Peak flux location
    total_l=np.sum(z)

    rad = np.linspace(5,np.shape(z)[0]-15,np.shape(z)[0])
    counts_in_r=np.zeros(len(rad))
    for ii,rr in enumerate(rad): 
      aperture = CircularAperture(centre, r=rr)

      #phot_table = aperture_photometry(counts_masked-sky, aperture)
      phot_table = aperture_photometry(z, aperture)
      counts_in_r[ii]=float(phot_table['aperture_sum'])

    #interpolate to find sub-pixel half-radius        
    f = interp1d(rad,counts_in_r, kind='cubic')
    xnew = np.linspace(np.min(rad), np.max(rad), num=80, endpoint=True)
    half_interp = xnew[np.argmax(f(xnew)>total_l/2)]
    """
    plt.figure()
    plt.plot(rad,counts_in_r,'o')
    plt.plot(xnew, f(xnew))
    plt.axvline(half_interp,color='r')
    plt.axhline(total_l/2)
    plt.show()
    """   
    return half_interp

def masking(counts,kernel):
    cent=[int(len(counts)/2)-1,int(len(counts)/2)-1]
    crop_cent=[int(len(counts)/2)-2,int(len(counts)/2)-2]
    crop_rad=5

     
    inner_ap = CircularAperture(crop_cent, r=crop_rad)
    inner_mask = inner_ap.to_mask(method='center').to_image(np.shape(counts))
    #inner_mask_shape = inner_mask.to_image((192,192))
    counts[inner_mask>0]=np.nan
    if kernel:
        # create a "fixed" image with NaNs replaced by interpolated values
        counts = interpolate_replace_nans(counts, kernel)
    else:
        ring_ap = CircularAnnulus(crop_cent, r_in=crop_rad,r_out=crop_rad+1)
        ring_dat=ring_ap.to_mask(method='center').multiply(counts)
        counts[np.isnan(counts)]=np.amin(ring_dat[ring_dat>0])

    return counts

if __name__=='__main__':
  indices=np.array([2,3,6,7,8,9,10,12,16,18,20,22,23,25,27,32,36,40,41,43,45,46,100])

  filt_no={'F115W':0,'F150W':1,'F200W':2,'F277W':3,'F356W':4,'F444W':5,'F560W':6,'F770W':7}
  wavelength={'F115W':1.15,'F150W':1.50,'F200W':2.00,'F277W':2.77,'F356W':3.56,'F444W':4.44}

  df=pd.read_pickle('/home/mmarshal/BLUETIDES/BlueTides/PIG_208/processed_data/quasarDatabase_processed.pkl')
  df=df[df['Sample']=='SDSS']

  rad_sersic={}
  err_sersic={}
  rad_sersic_true={}
  err_sersic_true={}

  fig,axes=plt.subplots(1,1,figsize=(3.4,2.8),gridspec_kw={'hspace':0.05,'left':0.2,'right':0.86,'top':0.93,'bottom':0.15})
  
  #for ff,filt in enumerate(['F115W','F150W','F200W','F277W','F356W','F444W']):#,'F560W','F770W']):
  for ff,filt in enumerate(['F200W']):#,'F356W','F444W']):#,'F560W','F770W']):
    rad_sersic[filt]=np.zeros(len(indices)-1) 
    err_sersic[filt]=np.zeros(len(indices)-1) 
    rad_sersic_true[filt]=np.zeros(len(indices)-1) 
    err_sersic_true[filt]=np.zeros(len(indices)-1) 
    
    print('______________________________'+filt+'________________________________')
    
    
    for ii,ind in enumerate(indices):
      if ind>41:
        ii-=1
      if ind!=41:
        #load images and flux values
        #ind=str(sys.argv[1])
        title='SDSS_'+str(ind)
        
        rad_sersic[filt][ii], err_sersic[filt][ii] = sersic_rad(filt,ind)
        rad_sersic_true[filt][ii], err_sersic_true[filt][ii] = sersic_rad_true(filt,ind)

    df_T = df[df[filt]==True]
    #axes.plot(df_T['Radius'],rad_sersic[filt][df[filt]==True],marker=markers[ff],label=filt,color='C'+str(ff),markerfacecolor='w',ls='',markeredgewidth=1.5)
    axes.errorbar(rad_sersic[filt][df[filt]==True],rad_sersic_true[filt][df[filt]==True],xerr=err_sersic[filt][df[filt]==True],marker=markers[ff],color='C'+str(ff),markerfacecolor='w',ls='',label=filt,markeredgewidth=1.5)


  axes.set_ylabel('New')
  axes.set_xlabel('Original')
    
  #to_plot = np.delete(indices,41)
  to_plot = np.array([2,3,6,7,8,9,10,12,16,18,20,22,23,25,27,32,36,40,43,45,46,100])

  fig = plt.figure(figsize=(10, 10))
  grid = ImageGrid(fig, 111, nrows_ncols=(int(np.ceil(len(to_plot)/2)), 4), axes_pad=0.1,
                     share_all=True, label_mode='L',
                     cbar_location='right', cbar_mode='single')
   
  ii=0 
  for quasar in to_plot:
        quasar = 'JWST_SDSS_' + str(quasar)
        plot_models(quasar, ii)
        ii+=1
   
  xy_format = plt.FormatStrFormatter(r'$%0.1f^{\prime\prime}$')
  for ax in grid:
        ax.set_xticks(_xytix)
        ax.set_yticks(_xytix)
        ax.xaxis.set_major_formatter(xy_format)
        ax.yaxis.set_major_formatter(xy_format)
  plt.show() 
