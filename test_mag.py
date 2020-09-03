import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from photutils import CircularAperture,CircularAnnulus
from photutils import aperture_photometry, background
from matplotlib.colors import SymLogNorm

if __name__=='__main__':
    #counts = 1/fits.getdata('/home/mmarshal/data_dragons/Observations/CFHQS-J0033-0125/ivm_CFHQS-J0033-0125_H.fits.gz')
    #counts = fits.getdata('/home/mmarshal/data_dragons/Observations/CFHQS-J0033-0125/sci_CFHQS-J0033-0125_H.fits.gz')
    #counts = 1/fits.getdata('/home/mmarshal/data_dragons/Observations/SDSS-J0203+0012/ivm_SDSS-J0203+0012_H.fits.gz')
    #counts = fits.getdata('/home/mmarshal/data_dragons/Observations/SDSS-J0203+0012/sci_SDSS-J0203+0012_H.fits.gz')
    counts = fits.getdata('data/sci_mock_JWST.fits')

    fig,ax=plt.subplots()
    im=ax.imshow((counts),norm=SymLogNorm(1e-7,vmin=np.amin(counts),vmax=np.amax(counts)))#norm=SymLogNorm(0.01,vmin=-0.2,vmax=0.2))#cmap='inferno',vmin=0,vmax=0.02)
      
    zeropoint=25.9463
    for ap in [5,10,40,50]:
      aperture = CircularAperture([len(counts)//2,len(counts)//2], r=ap)

      phot_table = aperture_photometry(counts, aperture)
      tot_counts=float(phot_table['aperture_sum'])
      print('Counts: ',tot_counts)
      mag=-2.5*np.log10(tot_counts)+zeropoint
      print('Mag: ',mag)
    
      annulus_masks = aperture.to_mask(method='center')
      annulus_data = annulus_masks[0].multiply(counts)

      circle2=plt.Circle([len(counts)//2,len(counts)//2],ap, fill=False,color='k',linestyle='--')
      ax.add_artist(circle2)
    plt.show()
   
 
