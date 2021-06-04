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


matplotlib.rcParams['font.size'] = (8)
#plt.rc('text', usetex=True)
plt.rc('font', family='serif')
markers= ['s','^','o','d','h','p']
_stretch = AsinhStretch()
_stretch.a = (0.01 - 0.0001)/2 / (0.01+0.0001)

def flux_to_mag(flux):
    return -2.5*np.log10(flux/(3631e9)) #flux in nJy

def full_mag(counts):
    return -2.5*np.log10(np.sum(counts))+25.9463

def sersic_mag(filt,ii):
    if filt=='F200W':
      _pattern = 'runJWST/SDSS_z7_SN/mcmc_out_mock_JWST_SDSS_{}_residual.fits'
      head = fits.getheader(_pattern.format(ii))
    else:
      _pattern = 'runJWST/SDSS_z7_{}/mcmc_out_mock_JWST_SDSS_{}_residual.fits'
      head = fits.getheader(_pattern.format(filt,ii))
    fit_mag=head['2SER_MAG'].split(' ')[0]
    err_mag=head['2SER_MAG'].split(' ')[-1]
    #fit_rad[jj]=head['2SER_RE'].split(' ')[0]
    #err_rad[jj]=head['2SER_RE'].split(' ')[-1]
    #fit_sers[jj]=head['2SER_N'].split(' ')[0]
    #err_sers[jj]=head['2SER_N'].split(' ')[-1]
    return fit_mag,err_mag
       
def ext_dict(dd):
    return np.concatenate(list(dd.values()))

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

  df=pd.read_pickle('/home/mmarshal/BLUETIDES/BlueTides/PIG_208/processed_data/quasarDatabase_processed.pkl')
  df=df[df['Sample']=='SDSS']

  mag_true={}
  mag_orig={}
  mag_sersic_fit={}
  err_sersic_fit={}
  mag_gauss={}
  mag_min={}

  #fig,axes=plt.subplots(2,3,gridspec_kw={'hspace':0.3},figsize=(8,4))
  #fig2,axes2=plt.subplots(2,3,gridspec_kw={'hspace':0.3},figsize=(8,4))
  #ax=axes.flatten()
  #ax2=axes2.flatten()
  fig3,axes3=plt.subplots(3,1,sharex=True,sharey=True,figsize=(3.2,6),gridspec_kw={'hspace':0.05,'bottom':0.1,'top':0.9})
  
  for ff,filt in enumerate(['F115W','F150W','F200W','F277W','F356W','F444W']):#,'F560W','F770W']):
  #for ff,filt in enumerate(['F277W','F356W','F444W']):#,'F560W','F770W']):
    mag_true[filt]=np.zeros(len(indices)-1) 
    mag_orig[filt]=np.zeros(len(indices)-1) 
    mag_gauss[filt]=np.zeros(len(indices)-1) 
    mag_min[filt]=np.zeros(len(indices)-1) 
    mag_sersic_fit[filt]=np.zeros(len(indices)-1) 
    err_sersic_fit[filt]=np.zeros(len(indices)-1) 

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

        mag_sersic_fit[filt][ii], err_sersic_fit[filt][ii] = sersic_mag(filt,ind)
    print('Sersic')
    #print(np.median(mag_sersic_fit[filt]-mag_true[filt]),np.max(mag_sersic_fit[filt]-mag_true[filt]),np.min(mag_sersic_fit[filt]-mag_true[filt]))
    print(np.percentile(mag_sersic_fit[filt][df[filt]==True]-mag_true[filt][df[filt]==True],[5,16,50,84,95]))
    print(np.percentile(mag_sersic_fit[filt][df[filt]==True]-mag_true[filt][df[filt]==True],[5,16,50,84,95])-np.median(mag_sersic_fit[filt][df[filt]==True]-mag_true[filt][df[filt]==True]))
    print('Original')
    #print(np.median(mag_orig[filt]-mag_true[filt]),np.max(mag_orig[filt]-mag_true[filt]),np.min(mag_orig[filt]-mag_true[filt]))
    print(np.percentile(mag_orig[filt][df[filt]==True]-mag_true[filt][df[filt]==True],[5,16,50,84,95]))
    print(np.percentile(mag_orig[filt][df[filt]==True]-mag_true[filt][df[filt]==True],[5,16,50,84,95])-np.median(mag_orig[filt][df[filt]==True]-mag_true[filt][df[filt]==True]))
    print('Gauss')
    #print(np.median(mag_gauss[filt]-mag_true[filt]),np.max(mag_gauss[filt]-mag_true[filt]),np.min(mag_gauss[filt]-mag_true[filt]))
    print(np.percentile(mag_gauss[filt][df[filt]==True]-mag_true[filt][df[filt]==True],[5,16,50,84,95]))
    print(np.percentile(mag_gauss[filt][df[filt]==True]-mag_true[filt][df[filt]==True],[5,16,50,84,95])-np.median(mag_gauss[filt][df[filt]==True]-mag_true[filt][df[filt]==True]))
  
    best_guess=mag_gauss[filt]
    up_err=mag_orig[filt]-best_guess
    low_err=best_guess-mag_min[filt]


    """    
    ax2[ff].errorbar(mag_true[filt],best_guess,yerr=[low_err,up_err],marker='o',ls='',label=filt+' Inc. Masking',color='C'+str(ff),markerfacecolor='w')
    ax2[ff].plot(mag_true[filt],mag_orig[filt],'s',label=filt+' No Masking',color='C'+str(ff),markerfacecolor='w')
    """ 

    ##Plot comparing true mphot to sersic fit, orig subtracted mphot and the best guess mphot after masking
    #axes3[0].plot(mag_true[filt],mag_sersic_fit[filt],marker=markers[ff],label=filt,color='C'+str(ff),markerfacecolor='w')
    axes3[0].errorbar(mag_true[filt][df[filt]==True],mag_sersic_fit[filt][df[filt]==True],yerr=err_sersic_fit[filt][df[filt]==True],marker=markers[ff],color='C'+str(ff),markerfacecolor='w',ls='',label=filt,markeredgewidth=1.5)
    axes3[1].plot(mag_true[filt][df[filt]==True],mag_orig[filt][df[filt]==True],marker=markers[ff],label=filt,color='C'+str(ff),markerfacecolor='w',ls='',markeredgewidth=1.5)
    axes3[2].errorbar(mag_true[filt][df[filt]==True],best_guess[df[filt]==True],yerr=[low_err[df[filt]==True],up_err[df[filt]==True]],marker=markers[ff],ls='',label=filt,color='C'+str(ff),markerfacecolor='w',markeredgewidth=1.5)

    """
    ax[ff].hist(best_guess-mag_true[filt],label='Inc. Masking',color='C'+str(ff),histtype='step')
    ax[ff].hist(mag_orig[filt]-mag_true[filt],label='No Masking',color='C'+str(ff),ls='--',histtype='step')
    ax[ff].set_title(filt)
    """
  for ff in range(0,3):
    axes3[ff].plot([22.5,27],[22.5,27],'k')
    axes3[ff].set_yticks([26,25,24,23])
    axes3[ff].axis('square')
    axes3[ff].set_xlim(26.2,22.8)
    axes3[ff].set_ylim(26.2,22.8)
  
  axes3[0].text(25,26.15,'Median: {0:.3f}'.format(np.median(ext_dict(mag_sersic_fit)-ext_dict(mag_true))))
  axes3[1].text(25.15,26.15,'Median:  {0:.3f}'.format(np.median(ext_dict(mag_orig)-ext_dict(mag_true))))
  axes3[2].text(24.95,26.15,'Median: {0:.3f}'.format(np.median(ext_dict(mag_gauss)-ext_dict(mag_true))))
  axes3[0].text(24.1,25.85,r'$\sigma$: {0:.3f}'.format(np.std(ext_dict(mag_sersic_fit)-ext_dict(mag_true))))
  axes3[1].text(24.1,25.85,r'$\sigma$: {0:.3f}'.format(np.std(ext_dict(mag_orig)-ext_dict(mag_true))))
  axes3[2].text(24.1,25.85,r'$\sigma$: {0:.3f}'.format(np.std(ext_dict(mag_gauss)-ext_dict(mag_true))))

  axes3[0].set_ylabel('Best-Fit\nSersic Magnitude')
  axes3[1].set_ylabel('Photometric Magnitude')
  axes3[2].set_ylabel('Masked\nPhotometric Magnitude')
  axes3[2].set_xlabel('True Host\nPhotometric Magnitude')
  axes3[0].legend(loc=(-0.5,1.05),ncol=3)
 
  """
  for ff in range(0,6):
     #ax2[ff].set_yscale('log')
     #ax2[ff].set_xscale('log')
    ax2[ff].plot([23,27],[23,27])
    ax2[ff].set_xlim(23.5,26.5)
    ax2[ff].set_ylim(23.5,26.5)
  """

#  plt.legend()
  #axes3[2].invert_xaxis()
  #axes3[2].invert_yaxis()
  plt.savefig('compare_magnitudes_blocking.pdf')
  plt.show()
