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


matplotlib.rcParams['font.size'] = (9)
#plt.rc('text', usetex=True)
plt.rc('font', family='serif')
markers= ['s','^','o','d','h','p']
_stretch = AsinhStretch()
_stretch.a = (0.01 - 0.0001)/2 / (0.01+0.0001)

def flux_to_mag(flux):
    return -2.5*np.log10(flux/(3631e9)) #flux in nJy

def full_mag(counts):
    return -2.5*np.log10(np.sum(counts))+25.9463
      

def plot_true(counts,ax):
    cent=[int(len(counts)/2),int(len(counts)/2)]
    ap = (len(counts)/2) #Full sized circular aperture for photometry.
    center = np.array(counts.shape)[::-1]/2
    extents = np.array([-center[0], center[0],
               -center[1], center[1]])*pxscale

    im=ax.imshow((counts),extent=extents,
                               origin='lower',
                               cmap='Spectral_r', norm=_pnorm,
                               interpolation='nearest')
    ax.axis([-0.6,0.6,-0.6,0.6])
    #plt.colorbar(im)


    circle2=plt.Circle(cent,ap, fill=False,color='k',linestyle='--')
    ax.add_artist(circle2)


    circle2=plt.Circle(crop_offset,crop_rad*pxscale, fill=False,color='k',linestyle='--')
    ax.add_artist(circle2)

    mag_true=ap_phot(counts,ax=False,kernel=False,mask=False,plot=False)
    ax.text(-0.38,-0.5,r'$m={}$'.format(np.round(mag_true,2)),color='w')
    
    return mag_true


def plot_orig(orig_counts,ax_orig):
      cent=[int(len(counts)/2),int(len(counts)/2)]
      ap = (len(counts)/2) #Full sized circular aperture for photometry.
      center = np.array(counts.shape)[::-1]/2
      extents = np.array([-center[0], center[0],
               -center[1], center[1]])*pxscale

      im=ax_orig.imshow((orig_counts),extent=extents,
                               origin='lower',
                               cmap='Spectral_r', norm=_pnorm,
                               interpolation='nearest')
      ax_orig.axis([-0.6,0.6,-0.6,0.6])

      circle2=plt.Circle(cent,ap, fill=False,color='k',linestyle='--')
      ax_orig.add_artist(circle2)


      circle2=plt.Circle(crop_offset,crop_rad*pxscale, fill=False,color='k',linestyle='--')
      ax_orig.add_artist(circle2)

      mag_orig=ap_phot(orig_counts,ax=False,kernel=False,mask=False,plot=False)
      ax_orig.text(-0.38,-0.5,r'$m={}$'.format(np.round(mag_orig,2)),color='w')
      return mag_orig


def ap_phot(counts,ax,kernel=False,mask=True,plot=True,level=False):
    cent=[int(len(counts)/2)-1,int(len(counts)/2)-1]

    ap = (len(counts)/2) #Full sized circular aperture for photometry.
    center = np.array(counts.shape)[::-1]/2

    #for ii,ap in enumerate(ap*np.linspace(0.05,1,20)):
    for ap in [ap]:#enumerate(np.linspace(1,35,10)):#[30]: Tested, brightness profiles flatten by r=25 (For 35 length array).
      aperture = CircularAperture(cent, r=ap)
      #print(4*FWHM/2,ap)
      #annulus_aperture = CircularAnnulus(cent, r_in=4*FWHM/2, r_out=ap)
      apers = [aperture]

     
      if mask:
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
          if level=='high':
            counts[np.isnan(counts)]=np.amax(ring_dat[ring_dat>0])
          elif level=='low':
            counts[np.isnan(counts)]=np.amin(ring_dat[ring_dat>0])
          else:
            counts[np.isnan(counts)]=np.median(ring_dat[ring_dat>0])
       
        
         


      #phot_table = aperture_photometry(counts, aperture)
      phot_table = aperture_photometry(counts, apers)
      ###No background level in mock images, don't need sky subtraction
      #bkg_mean = phot_table['aperture_sum_1'] / annulus_aperture.area
      #bkg_sum = bkg_mean * aperture.area
      ap_counts = phot_table['aperture_sum_0'] #- bkg_sum #Sky subtraction
      #print(bkg_sum,bkg_sum/phot_table['aperture_sum_0']) ##Bkg ~2% of total flux

      mag=-2.5*np.log10(ap_counts[0])+zeropoint
      #mag=-2.5*np.log10(ap_counts[ii])+zeropoint
      #print('Photometric measurement: {:04.2f}'.format(mag))
    
    """
    if ii==1:
      plot=True
      print(4*FWHM/2,ap)
    else:
      plot=False
    """
    if plot:    
      extents = np.array([-center[0], center[0],
               -center[1], center[1]])*pxscale
      _pnorm = ImageNormalize(vmin=-0.000005, vmax=0.01, stretch=_stretch, clip=True)#-0.000005, vmax=0.01
      center = np.array(counts.shape)[::-1]/2
      extents = np.array([-center[0], center[0],
               -center[1], center[1]])*pxscale
      im=ax.imshow((counts),extent=extents,
                               origin='lower',
                               cmap='Spectral_r', norm=_pnorm,
                               interpolation='nearest')
      ax.axis([-0.6,0.6,-0.6,0.6])
      #norm=_pnorm)#SymLogNorm(1e-1,vmin=np.amin(counts),vmax=np.amax(counts)),cmap='Spectral_r')#norm=SymLogNorm(0.01,vmin=-0.2,vmax=0.2))#cmap='inferno',vmin=0,vmax=0.02)
      cbar = plt.colorbar(im, cax=grid.cbar_axes[0])

      ax.text(-0.38,-0.5,r'$m={}$'.format(np.round(mag,2)),color='w')


      circle2=plt.Circle(cent,ap, fill=False,color='k',linestyle='--')
      ax.add_artist(circle2)


      circle2=plt.Circle(crop_offset,crop_rad*pxscale, fill=False,color='k',linestyle='--')
      ax.add_artist(circle2)
      #circle2=plt.Circle((0,0),2*FWHM/2, fill=False,color='k',linestyle='--')
      #ax.add_artist(circle2)
      #circle2=plt.Circle((0,0),3*FWHM/2, fill=False,color='k',linestyle='--')
      #ax.add_artist(circle2)

      #plt.ylim(len(counts)/3,len(counts)*2/3)
      #plt.xlim(len(counts)/3,len(counts)*2/3)
 
      #plt.plot(np.linspace(0.05,1,20),ap_counts) 
      #plt.axhline(np.sum(counts))
      #plt.axhline(0.99*np.sum(counts))
      #plt.ylim(0,np.sum(counts))
      #plt.show()
    return mag


if __name__=='__main__':
  indices=np.array([3,6,7,9,10])#2,3,6,7,8,9,10,12,16,18,20,22,23,25,27,32,36,40,41,43,45,46,100])

  filt_no={'F115W':0,'F150W':1,'F200W':2,'F277W':3,'F356W':4,'F444W':5,'F560W':6,'F770W':7}
  wavelength={'F115W':1.15,'F150W':1.50,'F200W':2.00,'F277W':2.77,'F356W':3.56,'F444W':4.44}

  #df=pd.read_pickle('/home/mmarshal/BLUETIDES/BlueTides/PIG_208/processed_data/quasarDatabase_processed.pkl')
  #df=df[df['Sample']=='SDSS']
  #print(df)
  host_mag={}
  #for ff,filt in enumerate(['F115W','F150W','F200W','F277W','F356W','F444W']):#,'F560W','F770W']):
  for ff,filt in enumerate(['F200W']):#,'F200W','F444W']):#,'F560W','F770W']):
    host_mag[ff]=np.zeros(len(indices)) 
    print('______________________________'+filt+'________________________________')
    
    zeropoint=25.9463
    if filt in ['F277W','F356W','F444W']:
        pxscale = 0.063/2 #arcsec
    else:
        pxscale = 0.031/2
    
    FWHM=0.2516*wavelength[filt]/6.57/pxscale
    crop_rad = 5 #1.5*FWHM
    print(crop_rad)
    _pnorm = ImageNormalize(vmin=-0.000005, vmax=0.01, stretch=_stretch, clip=True)#-0.000005, vmax=0.01

    fig=plt.figure(figsize=(6,7))
    grid = ImageGrid(fig, 111, nrows_ncols=(5, 4), axes_pad=0.1,
                     share_all=True, label_mode='L',
                     cbar_location='right', cbar_mode='single')

    kk=0
    for ii,ind in enumerate(indices):
      if ind>41:
        ii-=1
      if ind!=41:
        #load images and flux values
        #ind=str(sys.argv[1])
        title='SDSS_'+str(ind)
        
        if filt=='F200W':
          counts = fits.getdata('runJWST/SDSS_z7_SN/mcmc_out_mock_JWST_'+title+'_point_source_subtracted.fits')
        else:
          counts = fits.getdata('runJWST/SDSS_z7_'+filt+'/mcmc_out_mock_JWST_'+title+'_point_source_subtracted.fits')
        counts = gaussian_filter(counts, (1, 1))
        
        crop_cent=[int(len(counts)/2)-2,int(len(counts)/2)-2]
        crop_offset=(-1*pxscale,-1*pxscale)
    
        true_counts =  fits.getdata('data/sci_mock_JWST_{}_SDSS_{}_onlyHost.fits'.format(filt,ind))
        true_counts = gaussian_filter(true_counts, (1, 1))
        plot_true(true_counts,grid[3+kk])
        plot_orig(counts,grid[0+kk])
      
        #counts = fits.getdata('data/sci_mock_HST_f160w_'+title+'_onlyHost.fits')
        kernel = Gaussian2DKernel(x_stddev=FWHM)
        host_mag[ff][ii] = ap_phot(counts,grid[2+kk],kernel)

        host_mag[ff][ii]  = ap_phot(counts,grid[1+kk],kernel=False,level='low')
        
        #host_mag[ff][ii]  = ap_phot(counts,grid[4+kk],kernel=False,level='high')

        #kernel = Tophat2DKernel(radius=4*FWHM/2+2)
        #host_mag[ff][ii] = ap_phot(counts,grid[3+kk],kernel)
        
        #kernel = Box2DKernel(width=4*FWHM/2+4)
        #host_mag[ff][ii] = ap_phot(counts,grid[4],kernel)
        kk+=4

    grid[0].set_title('PSF-Subtracted',fontsize=9)
    grid[2].set_title('Masked:\nGaussian',fontsize=9)
    grid[1].set_title('Masked:\nMinimum',fontsize=9)
    grid[3].set_title('True Host',fontsize=9)

  """        
    plt.plot(host_mag[ff],host_mag_ann[ff],'o')
  plt.plot([23,26],[23,26],'k')
  plt.xlim(23.5,26.2)
  plt.ylim(23.5,26.2)
  #plt.axis('square')
  """
  
  plt.show()

