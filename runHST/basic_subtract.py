import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from photutils import CircularAperture,CircularAnnulus
from photutils import aperture_photometry, background
from matplotlib.colors import SymLogNorm

if __name__=='__main__':
    #counts = fits.getdata('mcmc_out_mock_HST_convolved_model.fits')
    counts = fits.getdata('../data/sci_mock_HST.fits')

    fig,ax=plt.subplots()
    im=ax.imshow((counts),norm=SymLogNorm(1e-7,vmin=np.amin(counts),vmax=np.amax(counts)))#norm=SymLogNorm(0.01,vmin=-0.2,vmax=0.2))#cmap='inferno',vmin=0,vmax=0.02)
    
    zeropoint=25.9463
    for ap in [30,40,50,60]:
      aperture = CircularAperture([59.5,59.5], r=ap)

      phot_table = aperture_photometry(counts, aperture)
      tot_counts=float(phot_table['aperture_sum'])
      mag=-2.5*np.log10(tot_counts)+zeropoint
      print(mag)
    
      annulus_masks = aperture.to_mask(method='center')
      annulus_data = annulus_masks[0].multiply(counts)

      circle2=plt.Circle([59.5,59.5],ap, fill=False,color='k',linestyle='--')
      ax.add_artist(circle2)
    plt.show()
   
 
