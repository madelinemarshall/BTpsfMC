import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from photutils import CircularAperture,CircularAnnulus
from photutils import aperture_photometry, background
from matplotlib.colors import SymLogNorm

if __name__=='__main__':
    ivm = fits.getdata('/home/mmarshal/data_dragons/Observations/CFHQS-J0033-0125/ivm_CFHQS-J0033-0125_H.fits.gz')[45:80,45:80]
    sci = fits.getdata('/home/mmarshal/data_dragons/Observations/CFHQS-J0033-0125/sci_CFHQS-J0033-0125_H.fits.gz')[45:80,45:80]
    sqrtsci = 1/np.abs(sci)

    fig,ax=plt.subplots(1,3,sharex=True,sharey=True)
    im=ax[0].imshow((ivm),norm=SymLogNorm(1e-7,vmin=np.amin(sqrtsci),vmax=np.amax(ivm)))#norm=SymLogNorm(0.01,vmin=-0.2,vmax=0.2))#cmap='inferno',vmin=0,vmax=0.02)
    im=ax[1].imshow((sqrtsci),norm=SymLogNorm(1e-7,vmin=np.amin(sqrtsci),vmax=np.amax(ivm)))#norm=SymLogNorm(0.01,vmin=-0.2,vmax=0.2))#cmap='inferno',vmin=0,vmax=0.02)
    im=ax[2].imshow((ivm/sqrtsci))#,norm=SymLogNorm(1e-7,vmin=np.amin(sqrtsci),vmax=np.amax(ivm)))#norm=SymLogNorm(0.01,vmin=-0.2,vmax=0.2))#cmap='inferno',vmin=0,vmax=0.02)
    plt.colorbar(im)
   
    ax[0].set_title('ivm')
    ax[1].set_title('1/sci')
    ax[2].set_title('ivm*sci')
    plt.show()
   
 
    ivm = fits.getdata('data/sci_mock_HST.fits')
    sci = fits.getdata('data/ivm_mock_HST.fits')
    sqrtsci = 1/np.abs(sci)

    fig,ax=plt.subplots(1,3,sharex=True,sharey=True)
    im=ax[0].imshow((ivm),norm=SymLogNorm(1e-7,vmin=np.amin(ivm),vmax=np.amax(sqrtsci)))#norm=SymLogNorm(0.01,vmin=-0.2,vmax=0.2))#cmap='inferno',vmin=0,vmax=0.02)
    im=ax[1].imshow((sqrtsci),norm=SymLogNorm(1e-7,vmin=np.amin(ivm),vmax=np.amax(sqrtsci)))#norm=SymLogNorm(0.01,vmin=-0.2,vmax=0.2))#cmap='inferno',vmin=0,vmax=0.02)
    im=ax[2].imshow((ivm/sqrtsci))#,norm=SymLogNorm(1e-7,vmin=np.amin(sqrtsci),vmax=np.amax(ivm)))#norm=SymLogNorm(0.01,vmin=-0.2,vmax=0.2))#cmap='inferno',vmin=0,vmax=0.02)
    plt.colorbar(im)
   
    ax[0].set_title('ivm')
    ax[1].set_title('1/sci')
    ax[2].set_title('ivm*sci')
    plt.show()
