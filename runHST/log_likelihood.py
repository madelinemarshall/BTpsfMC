import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from photutils import CircularAperture,CircularAnnulus
from photutils import aperture_photometry, background
from matplotlib.colors import SymLogNorm, LogNorm
from psfMC.utils import norm_psf
from scipy.ndimage import convolve

if __name__=='__main__':
    ivm_in = fits.getdata('../data/ivm_mock_HST.fits')
    obs_var=1/ivm_in 
    psf_orig = fits.getdata('../data/sci_PSF_HST.fits')
    ivm_psf_orig = fits.getdata('../data/ivm_PSF_HST.fits')
    
    badpx = ~np.isfinite(psf_orig) | ~np.isfinite(ivm_psf_orig) | (ivm_psf_orig <= 0)
    psf_orig[badpx] = 0
    ivm_psf_orig[badpx] = 0
    psf,ivm_psf=norm_psf(psf_orig,ivm_psf_orig)
    psf_var = np.where(ivm_psf <= 0, 0, 1 / ivm_psf)

    #for dirc in ['realObs_fakePSF/','fakeObs_realPSF/','croppedTrueSolution/','']:
    for dirc in ['croppedUnconstrained/','croppedTrueSolution/','']:
      print(dirc)
      resid = fits.getdata(dirc+'mcmc_out_mock_HST_residual.fits')
      #psf = fits.getdata('../data/sci_PSF_HST.fits')
      ivm = fits.getdata(dirc+'mcmc_out_mock_HST_composite_ivm.fits')
      conv_px = fits.getdata(dirc+'mcmc_out_mock_HST_convolved_model.fits')
      raw_px = fits.getdata(dirc+'mcmc_out_mock_HST_raw_model.fits')

      
      ll=-0.5*np.sum(resid**2*ivm-np.log10(0.5/np.pi*ivm))
      print(ll) 
      ll=-0.5*np.sum(resid**2*ivm)
      print(ll) 
      ll=-0.5*np.sum(np.log10(0.5/np.pi*ivm))
      print(ll) 

      #ll=-0.5*np.sum(resid**2*ivm_in-np.log10(0.5/np.pi*ivm_in))
      #print(ll) 
      #ll=-0.5*np.sum(resid**2*ivm_in)
      #print(ll) 
      #ll=-0.5*np.sum(np.log10(0.5/np.pi*ivm_in))
      #print(ll) 
 
      #print(np.shape(np.fft.rfft2(raw_px**2,[70,70])))
      #print(np.shape(np.dot(np.fft.rfft2(raw_px),np.fft.rfft2(psf_var))))
      #print(np.shape(psf_var))
      #print(np.shape(psf))
      model_var=convolve(raw_px**2,psf_var)
      comp_ivm = 1/(model_var+obs_var)

    if False: 
      dirc='/home/mmarshal/data_dragons/Observations/CFHQS-J0033-0125/'
      print(dirc)
      resid = fits.getdata(dirc+'out_CFHQS-J0033-0125_H_residual.fits')
      #psf = fits.getdata('../data/sci_PSF_HST.fits')
      ivm = fits.getdata(dirc+'out_CFHQS-J0033-0125_H_composite_ivm.fits')
      conv_px = fits.getdata(dirc+'out_CFHQS-J0033-0125_H_convolved_model.fits')
      raw_px = fits.getdata(dirc+'out_CFHQS-J0033-0125_H_raw_model.fits')
      
      ll=-0.5*np.sum(resid**2*ivm-np.log10(0.5/np.pi*ivm))
      print(ll) 
      ll=-0.5*np.sum(resid**2*ivm)
      print(ll) 
      ll=-0.5*np.sum(np.log10(0.5/np.pi*ivm))
      print(ll) 
    
    if False:
      ll=-0.5*(resid**2*ivm-np.log10(0.5/np.pi*ivm))
      fig,ax=plt.subplots()
      im=ax.imshow(ll,norm=SymLogNorm(1e-7,vmin=np.amin(ll),vmax=np.amax(ll)))#norm=SymLogNorm(0.01,vmin=-0.2,vmax=0.2))#cmap='inferno',vmin=0,vmax=0.02)
      plt.show()
    
    if False:
      fig,ax=plt.subplots(1,3)   
      #im=ax[2].imshow(psf_orig-1/ivm_psf_orig,norm=SymLogNorm(1e-7,vmin=np.amin(psf_orig),vmax=np.amax(psf_orig)))#norm=SymLogNorm(0.01,vmin=-0.2,vmax=0.2))#cmap='inferno',vmin=0,vmax=0.02)
      #im=ax[1].imshow(1/ivm_psf_orig,norm=SymLogNorm(1e-7,vmin=np.amin(psf_orig),vmax=np.amax(psf_orig)))#norm=SymLogNorm(0.01,vmin=-0.2,vmax=0.2))#cmap='inferno',vmin=0,vmax=0.02)
      #im=ax[0].imshow(psf_orig,norm=SymLogNorm(1e-7,vmin=np.amin(psf_orig),vmax=np.amax(psf_orig)))#norm=SymLogNorm(0.01,vmin=-0.2,vmax=0.2))#cmap='inferno',vmin=0,vmax=0.02)
      im=ax[2].imshow(psf-1/ivm_psf,norm=SymLogNorm(1e-7,vmin=np.amin(psf),vmax=np.amax(psf)))#norm=SymLogNorm(0.01,vmin=-0.2,vmax=0.2))#cmap='inferno',vmin=0,vmax=0.02)
      im=ax[1].imshow(1/ivm_psf,norm=SymLogNorm(1e-7,vmin=np.amin(psf),vmax=np.amax(psf)))#norm=SymLogNorm(0.01,vmin=-0.2,vmax=0.2))#cmap='inferno',vmin=0,vmax=0.02)
      im=ax[0].imshow(psf,norm=SymLogNorm(1e-7,vmin=np.amin(psf),vmax=np.amax(psf)))#norm=SymLogNorm(0.01,vmin=-0.2,vmax=0.2))#cmap='inferno',vmin=0,vmax=0.02)
      plt.show()

    if True:
      fig,ax=plt.subplots(2,3)
      im=ax[0,0].imshow(ivm,norm=LogNorm(vmin=np.amin(ivm),vmax=np.amax(ivm)))#norm=SymLogNorm(0.01,vmin=-0.2,vmax=0.2))#cmap='inferno',vmin=0,vmax=0.02)
      im=ax[0,1].imshow(comp_ivm,norm=LogNorm(vmin=np.amin(ivm),vmax=np.amax(ivm)))#norm=SymLogNorm(0.01,vmin=-0.2,vmax=0.2))#cmap='inferno',vmin=0,vmax=0.02)
      im=ax[0,2].imshow(1/model_var,norm=LogNorm(vmin=np.amin(ivm),vmax=np.amax(ivm)))#norm=SymLogNorm(0.01,vmin=-0.2,vmax=0.2))#cmap='inferno',vmin=0,vmax=0.02)
      im=ax[1,0].imshow(1/obs_var,norm=LogNorm(vmin=np.amin(ivm),vmax=np.amax(ivm)))#norm=SymLogNorm(0.01,vmin=-0.2,vmax=0.2))#cmap='inferno',vmin=0,vmax=0.02)
      im=ax[1,1].imshow(ivm_psf,norm=LogNorm(vmin=np.amin(ivm),vmax=np.amax(ivm)))#norm=SymLogNorm(0.01,vmin=-0.2,vmax=0.2))#cmap='inferno',vmin=0,vmax=0.02)
      im=ax[1,2].imshow(comp_ivm-ivm,norm=LogNorm(vmin=np.amin(ivm),vmax=np.amax(ivm)))#norm=SymLogNorm(0.01,vmin=-0.2,vmax=0.2))#cmap='inferno',vmin=0,vmax=0.02)
      plt.show()

