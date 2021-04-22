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
      

def ap_phot(counts,kernel=False,mask=True,level=False):
    cent=[int(len(counts)/2)-1,int(len(counts)/2)-1]
    crop_cent=[int(len(counts)/2)-2,int(len(counts)/2)-2]

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
    
    return mag


if __name__=='__main__':
  indices=np.array([2,3,6,7,8,9,10,12,16,18,20,22,23,25,27,32,36,40,41,43,45,46,100])

  filt_no={'F115W':0,'F150W':1,'F200W':2,'F277W':3,'F356W':4,'F444W':5,'F560W':6,'F770W':7}
  wavelength={'F115W':1.15,'F150W':1.50,'F200W':2.00,'F277W':2.77,'F356W':3.56,'F444W':4.44}

  mag_true={}
  mag_orig={}
  mag_gauss={}
  mag_min={}

  fig,axes=plt.subplots(2,3,gridspec_kw={'hspace':0.3},figsize=(8,4))
  fig2,axes2=plt.subplots(2,3,gridspec_kw={'hspace':0.3},figsize=(8,4))
  ax=axes.flatten()
  ax2=axes2.flatten()
  for ff,filt in enumerate(['F115W','F150W','F200W','F277W','F356W','F444W']):#,'F560W','F770W']):
  #for ff,filt in enumerate(['F150W']):#,'F200W','F444W']):#,'F560W','F770W']):
    mag_true[filt]=np.zeros(len(indices)-1) 
    mag_orig[filt]=np.zeros(len(indices)-1) 
    mag_gauss[filt]=np.zeros(len(indices)-1) 
    mag_min[filt]=np.zeros(len(indices)-1) 

    print('______________________________'+filt+'________________________________')
    
    zeropoint=25.9463
    if filt in ['F277W','F356W','F444W']:
        pxscale = 0.063/2 #arcsec
    else:
        pxscale = 0.031/2
    
    FWHM=0.2516*wavelength[filt]/6.57/pxscale
    crop_rad = 5
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
        
    
        true_counts =  fits.getdata('data/sci_mock_JWST_{}_SDSS_{}_onlyHost.fits'.format(filt,ind))
        true_counts = gaussian_filter(true_counts, (1, 1))

        mag_true[filt][ii]=ap_phot(true_counts,kernel=False,mask=False)
        mag_orig[filt][ii]=ap_phot(counts,kernel=False,mask=False)

      
        #counts = fits.getdata('data/sci_mock_HST_f160w_'+title+'_onlyHost.fits')
        kernel = Gaussian2DKernel(x_stddev=FWHM)
        mag_gauss[filt][ii] = ap_phot(counts,kernel)

        mag_min[filt][ii]  = ap_phot(counts,kernel=False,level='low')

    print('Original')
    print(np.median(mag_orig[filt]-mag_true[filt]),np.max(mag_orig[filt]-mag_true[filt]),np.min(mag_orig[filt]-mag_true[filt]))
    print('Gauss')
    print(np.median(mag_gauss[filt]-mag_true[filt]),np.max(mag_gauss[filt]-mag_true[filt]),np.min(mag_gauss[filt]-mag_true[filt]))
  
    best_guess=mag_gauss[filt]
    up_err=mag_orig[filt]-best_guess
    low_err=best_guess-mag_min[filt]

    #if filt in ['F115W','F150W']:
    #  ax[0].errorbar(mag_true[filt],best_guess,yerr=[low_err,up_err],marker='o',ls='',label=filt+' Inc. Masking',color='C'+str(ff),markerfacecolor='w')
    #  ax[0].plot(mag_true[filt],mag_orig[filt],'s',label=filt+' No Masking',color='C'+str(ff),markerfacecolor='w')
    #elif filt in ['F200W','F277W']:
    #  ax[1].errorbar(mag_true[filt],best_guess,yerr=[low_err,up_err],marker='o',ls='',label=filt+' Inc. Masking',color='C'+str(ff),markerfacecolor='w')
    #  ax[1].plot(mag_true[filt],mag_orig[filt],'s',label=filt+' No Masking',color='C'+str(ff),markerfacecolor='w')
    
    ax2[ff].errorbar(mag_true[filt],best_guess,yerr=[low_err,up_err],marker='o',ls='',label=filt+' Inc. Masking',color='C'+str(ff),markerfacecolor='w')
    ax2[ff].plot(mag_true[filt],mag_orig[filt],'s',label=filt+' No Masking',color='C'+str(ff),markerfacecolor='w')
   # ax2[ff].errorbar(10**(-0.4*mag_true[filt]),10**(-0.4*best_guess),yerr=[low_err,up_err],marker='o',ls='',label=filt+' Inc. Masking',color='C'+str(ff),markerfacecolor='w')
    
    #ax2[ff].plot(mag_true[filt],10**(-0.4*best_guess)-10**(-0.4*mag_true[filt]),marker='o',ls='',label=filt+' Inc. Masking',color='C'+str(ff),markerfacecolor='w')
    #ax2[ff].plot(mag_true[filt],10**(-0.4*mag_orig[filt])-10**(-0.4*mag_true[filt]),'s',label=filt+' No Masking',color='C'+str(ff),markerfacecolor='w')
    
    ax[ff].hist(best_guess-mag_true[filt],label='Inc. Masking',color='C'+str(ff),histtype='step')
    ax[ff].hist(mag_orig[filt]-mag_true[filt],label='No Masking',color='C'+str(ff),ls='--',histtype='step')
    ax[ff].set_title(filt)
    ax[ff].set_xlim(-1,0.75)
    #if ff>2:
    #  ax[ff].set_xlabel('True Host Photometric Magnitude')
    #if ff==0 or ff==3:
    #  ax[ff].set_ylabel('Magnitude Estimate')

    #else:
    #ax[0].plot(mag_true[filt],mag_orig[filt],'o',label=filt+' No Masking',color='C'+str(ff))
  print(mag_true)
  print(mag_orig)
  print(mag_gauss)
  print(mag_min)



  for ff in range(0,6):
     #ax2[ff].set_yscale('log')
     #ax2[ff].set_xscale('log')
    ax2[ff].plot([23,27],[23,27])
    ax2[ff].set_xlim(23.5,26.5)
    ax2[ff].set_ylim(23.5,26.5)
  plt.legend()
  plt.show()
