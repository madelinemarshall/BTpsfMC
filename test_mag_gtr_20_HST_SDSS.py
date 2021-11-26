import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from photutils import CircularAperture,CircularAnnulus
from photutils import aperture_photometry, background
from matplotlib.colors import SymLogNorm
import sys

if __name__=='__main__':
    #counts = 1/fits.getdata('/home/mmarshal/data_dragons/Observations/CFHQS-J0033-0125/ivm_CFHQS-J0033-0125_H.fits.gz')
    #counts = fits.getdata('/home/mmarshal/data_dragons/Observations/CFHQS-J0033-0125/sci_CFHQS-J0033-0125_H.fits.gz')
    #counts = 1/fits.getdata('/home/mmarshal/data_dragons/Observations/SDSS-J0203+0012/ivm_SDSS-J0203+0012_H.fits.gz')
    #counts = fits.getdata('/home/mmarshal/data_dragons/Observations/SDSS-J0203+0012/sci_psf_02hr_H.fits.gz')
    #counts = fits.getdata('/home/mmarshal/data_dragons/Observations/SDSS-J2054-0005/sci_psf_21hr_H.fits.gz')

    #fig,ax=plt.subplots()
    #im=ax.imshow((counts),norm=SymLogNorm(1e-7,vmin=np.amin(counts),vmax=np.amax(counts)))#norm=SymLogNorm(0.01,vmin=-0.2,vmax=0.2))#cmap='inferno',vmin=0,vmax=0.02)
      
    zeropoint=25.9463

    for ind in [2,   3,   6,   7,   8,   9,  10,  12,  16, 18,  20,  22,  23,  25,  27,  32,  36,  40,  43,  45,  46, 100]:
      counts = fits.getdata('data/sci_mock_HST_f160w_SDSS_{}.fits'.format(ind))
      for ap in [30]:
        aperture = CircularAperture([len(counts)//2,len(counts)//2], r=ap)

        phot_table = aperture_photometry(counts, aperture)
        tot_counts=float(phot_table['aperture_sum'])
        mag=-2.5*np.log10(tot_counts)+zeropoint
        if mag<20.5:
          print('Index: ',ind,'Mag: ',mag)
    
        #annulus_masks = aperture.to_mask(method='center')
        #annulus_data = annulus_masks[0].multiply(counts)

      #circle2=plt.Circle([len(counts)//2,len(counts)//2],ap, fill=False,color='k',linestyle='--')
      #ax.add_artist(circle2)
    #plt.show()
   
 
