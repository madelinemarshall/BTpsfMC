import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from photutils import CircularAperture,CircularAnnulus
from photutils import aperture_photometry, background
from matplotlib.colors import SymLogNorm

if __name__=='__main__':
    sci = fits.getdata('../data/sci_mock_HST.fits')
    #psf = fits.getdata('../data/sci_PSF_HST.fits')
    psf = fits.getdata('mcmc_out_mock_HST_convolved_model.fits')

    fig,ax=plt.subplots(1,3,sharex=True,sharey=True)
    im=ax[0].imshow((sci),norm=SymLogNorm(1e-7,vmin=np.amin(sci),vmax=np.amax(sci)))#norm=SymLogNorm(0.01,vmin=-0.2,vmax=0.2))#cmap='inferno',vmin=0,vmax=0.02)
    im=ax[1].imshow((psf),norm=SymLogNorm(1e-7,vmin=np.amin(sci),vmax=np.amax(sci)))#norm=SymLogNorm(0.01,vmin=-0.2,vmax=0.2))#cmap='inferno',vmin=0,vmax=0.02)
    im=ax[2].imshow((sci-psf),norm=SymLogNorm(1e-7,vmin=np.amin(sci),vmax=np.amax(sci)))#norm=SymLogNorm(0.01,vmin=-0.2,vmax=0.2))#cmap='inferno',vmin=0,vmax=0.02)
    
    plt.show()
   
 
