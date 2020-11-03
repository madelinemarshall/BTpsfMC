import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from photutils import CircularAperture,CircularAnnulus
from photutils import aperture_photometry, background
from matplotlib.colors import SymLogNorm
import sys
import pandas as pd

if __name__=='__main__':

    title='SDSS_'+str(sys.argv[1])
    #counts = fits.getdata('mcmc_out_mock_HST_convolved_model.fits')
    counts = fits.getdata('data/sci_mock_HST_f160w_'+title+'_onlyHost.fits')
    counts = fits.getdata('data/sci_mock_HST_f160w_'+title+'.fits')
    #counts = fits.getdata('testQuasarSamples_onlyHost/sci_mock_JWST_F200W_MMBH_onlyHost.fits')
    head = fits.getheader('runHST/SDSS_z7/mcmc_out_mock_HST_'+title+'_residual.fits')
    print('Host subtraction fit: ',head['2SER_MAG'])

    


    fig,ax=plt.subplots()
    im=ax.imshow((counts),norm=SymLogNorm(1e-7,vmin=np.amin(counts),vmax=np.amax(counts)))#norm=SymLogNorm(0.01,vmin=-0.2,vmax=0.2))#cmap='inferno',vmin=0,vmax=0.02)
    
    cent=[int(len(counts)/2),int(len(counts)/2)]
    zeropoint=25.9463
    for ap in [10,20,30]:
      aperture = CircularAperture(cent, r=ap)

      phot_table = aperture_photometry(counts, aperture)
      tot_counts=float(phot_table['aperture_sum'])
      mag=-2.5*np.log10(tot_counts)+zeropoint
      print('Photometric measurement: {:04.2f}'.format(mag))
    
      annulus_masks = aperture.to_mask(method='center')
      annulus_data = annulus_masks[0].multiply(counts)

      circle2=plt.Circle(cent,ap, fill=False,color='k',linestyle='--')
      ax.add_artist(circle2)
    #plt.show()
   
 
    print('Quasar: Host subtraction fit: ',head['1PS_MAG'])

