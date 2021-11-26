import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import fits
import scipy.ndimage as ndi
from astropy.visualization import simple_norm
from astropy.modeling import models
from astropy.convolution import convolve
import photutils
import time
import statmorph
from statmorph.utils.image_diagnostics import make_figure
from scipy.ndimage import gaussian_filter
from astropy.convolution import Gaussian2DKernel, Tophat2DKernel, Box2DKernel, interpolate_replace_nans
from photutils import CircularAperture,CircularAnnulus


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


def perform_statmorph(image,ivm):
  var=1/ivm
  std = np.sqrt(var)
  print(std)

  threshold = photutils.detect_threshold(image, 1.5)
  npixels = 5  # minimum number of connected pixels
  segm = photutils.detect_sources(image, threshold, npixels)


  # Keep only the largest segment
  label = np.argmax(segm.areas) + 1
  segmap_or = segm.data == label

  segmap_float = ndi.uniform_filter(np.float64(segmap_or), size=10)
  segmap = segmap_float > 0.5

  start = time.time()
  source_morphs = statmorph.source_morphology(
    #image, segmap, gain = 10000)#, psf=psf)
    image, segmap, weightmap=std,verbose=True,segmap_overlap_ratio=0.6)#,mask=np.invert(segmap))# psf=psf)
  print('Time: %g s.' % (time.time() - start))

  morph = source_morphs[0]

  print('rhalf_circ =', morph.rhalf_circ,'rhalf_ellip =', morph.rhalf_ellip)
  print('r20 =', morph.r20,'r80 =', morph.r80)
  print('Gini =', morph.gini,'M20 =', morph.m20)
  #print('F(G, M20) =', morph.gini_m20_bulge)
  #print('S(G, M20) =', morph.gini_m20_merger)
  print('sn_per_pixel =', morph.sn_per_pixel)
  print('C =', morph.concentration,'A =', morph.asymmetry,'S =', morph.smoothness)
  print('sersic_amplitude =', morph.sersic_amplitude,'sersic_rhalf =', morph.sersic_rhalf,'sersic_n =', morph.sersic_n)
  #print('sersic_xc =', morph.sersic_xc)
  #print('sersic_yc =', morph.sersic_yc)
  #print('sersic_ellip =', morph.sersic_ellip)
  #print('sersic_theta =', morph.sersic_theta)
  #print('sky_mean =', morph.sky_mean)
  #print('sky_median =', morph.sky_median)
  #print('sky_sigma =', morph.sky_sigma)
  print('flag =', morph.flag)
  print('flag_sersic =', morph.flag_sersic)

  
  ny, nx = image.shape
  y, x = np.mgrid[0:ny, 0:nx]
  fitted_model = statmorph.ConvolvedSersic2D(
    amplitude=morph.sersic_amplitude,
    r_eff=morph.sersic_rhalf,
    n=morph.sersic_n,
    x_0=morph.sersic_xc,
    y_0=morph.sersic_yc,
    ellip=morph.sersic_ellip,
    theta=morph.sersic_theta)
  #fitted_model.set_psf(psf)  # required when using ConvolvedSersic2D
  #image_model = fitted_model(x, y)
  #bg_noise = (1.0 / snp) * np.random.standard_normal(size=(ny, nx))
  fig,ax = plt.subplots(1,3,figsize=(15,5),sharex=True,sharey=True)
  ax[0].imshow(image, cmap='gray', origin='lower',
           norm=simple_norm(image, stretch='log', log_a=1000))
  ax[0].set_title('Original image')
  #ax.imshow(image_model + bg_noise, cmap='gray', origin='lower',
  #         norm=simple_norm(image, stretch='log', log_a=10000))
  #ax.set_title('Fitted model')
  ax[1].imshow(segmap_or, cmap='gray', origin='lower',
           norm=simple_norm(image, stretch='log', log_a=10000))
  ax[1].set_title('Segmap')
  ax[2].imshow(segmap, cmap='gray', origin='lower',
           norm=simple_norm(image, stretch='log', log_a=10000))
  ax[2].set_title('Segmap')
  #residual = image - image_model
  #ax.imshow(residual, cmap='gray', origin='lower',
  #         norm=simple_norm(residual, stretch='linear'))
  ax[2].set_title('Residual')

  fig = make_figure(morph)
  
  return

if __name__=='__main__':
  indices=np.array([12])#,16,20,22,40,100])#2,3,6,7,8,9,10,12,16,18,20,22,23,25,27,32,36,40,41,43,45,46,100])

  filt_no={'F115W':0,'F150W':1,'F200W':2,'F277W':3,'F356W':4,'F444W':5,'F560W':6,'F770W':7}
  wavelength={'F115W':1.15,'F150W':1.50,'F200W':2.00,'F277W':2.77,'F356W':3.56,'F444W':4.44}

  df=pd.read_pickle('/home/mmarshal/BLUETIDES/BlueTides/PIG_208/processed_data/quasarDatabase_processed.pkl')
  df=df[df['Sample']=='SDSS']

  #fig,axes=plt.subplots(2,3,gridspec_kw={'hspace':0.3},figsize=(8,4))
  #fig2,axes2=plt.subplots(2,3,gridspec_kw={'hspace':0.3},figsize=(8,4))
  #ax=axes.flatten()
  #ax2=axes2.flatten()
  #fig3,axes3=plt.subplots(3,1,sharex=True,sharey=True,figsize=(3.4,8),gridspec_kw={'hspace':0.05,'bottom':0.07,'top':0.93})
  
  #for ff,filt in enumerate(['F115W','F150W','F200W','F277W','F356W','F444W']):#,'F560W','F770W']):
  for ff,filt in enumerate(['F200W']):#,'F356W','F444W']):#,'F560W','F770W']):

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
          ivm = fits.getdata('runJWST/SDSS_z7_SN/mcmc_out_mock_JWST_'+title+'_composite_ivm.fits')
          _pattern = 'runJWST/SDSS_z7_SN/mcmc_out_mock_JWST_SDSS_{}_residual.fits'
          _pattern_OH= 'runJWST/SDSS_z7_SN_onlyHost/mcmc_out_mock_JWST_SDSS_{}_residual.fits'
          head_OH = fits.getheader(_pattern_OH.format(ind))
          print('Fit only host: Ser mag {}, Ser Re {}, Ser n {}'.format(head_OH['1SER_MAG'],head_OH['1SER_RE'],head_OH['1SER_N']))
        else:
          counts = fits.getdata('runJWST/SDSS_z7_'+filt+'/mcmc_out_mock_JWST_'+title+'_point_source_subtracted.fits')
          ivm = fits.getdata('runJWST/SDSS_z7_'+filt+'/mcmc_out_mock_JWST_'+title+'_composite_ivm.fits')
          _pattern = 'runJWST/SDSS_z7_'+filt+'/mcmc_out_mock_JWST_'+title+'_residual.fits'
        psf = fits.open('data/sci_PSF_JWST_{}_SN.fits'.format(filt))[0].data
        head = fits.getheader(_pattern.format(ind))
        print('Fit: Ser mag {}, Ser Re {}, Ser n {}'.format(head['2SER_MAG'],head['2SER_RE'],head['2SER_N']))

        counts = gaussian_filter(counts, (1, 1))
        
        counts_gauss = np.copy(counts)
        counts_gauss = masking(counts_gauss,kernel = Gaussian2DKernel(x_stddev=FWHM))
    
        true_counts =  fits.getdata('data/sci_mock_JWST_{}_SDSS_{}_onlyHost.fits'.format(filt,ind))
        #true_counts =  fits.getdata('data/sci_mock_JWST_{}_SDSS_{}_onlyHost_noNoise.fits'.format(filt,ind))
        true_ivm =  fits.getdata('data/ivm_mock_JWST_{}_SDSS_{}_onlyHost.fits'.format(filt,ind))
        #true_counts = gaussian_filter(true_counts, (1, 1))

        true_gauss = np.copy(true_counts)
        true_gauss = masking(true_gauss,kernel = Gaussian2DKernel(x_stddev=FWHM))
        
        perform_statmorph(true_counts,true_ivm)
        #perform_statmorph(true_gauss,true_ivm)
        #perform_statmorph(counts,ivm)
        #perform_statmorph(counts_gauss,ivm)
  plt.show()
