import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from photutils import CircularAperture,CircularAnnulus
from photutils import aperture_photometry, background
from matplotlib.colors import SymLogNorm
import sys
import pandas as pd

if __name__=='__main__':

    if len(sys.argv)>1:
      title=sys.argv[1]
    else:
      title='MMBH'
    #counts = fits.getdata('mcmc_out_mock_HST_convolved_model.fits')
    counts = fits.getdata('../data/sci_mock_JWST_F200W_'+title+'_onlyHost.fits')
    #counts = fits.getdata('testQuasarSamples_onlyHost/sci_mock_JWST_F200W_MMBH_onlyHost.fits')
    head = fits.getheader('testQuasarSamples_onlyHost/mcmc_out_mock_JWST_'+title+'_residual.fits')
    print('True host fit: ',head['1SER_MAG'])
    #position=head['1SER_XY'].split(' ')[0].replace('(','').replace(')','').split(',')
    #for ii,el in enumerate(position):
    #  position[ii]=float(el)
    head = fits.getheader('testQuasarSamples_host/mcmc_out_mock_JWST_'+title+'_residual.fits')
    print('Subtraction fit: ',head['2SER_MAG'])

    


    fig,ax=plt.subplots()
    im=ax.imshow((counts),norm=SymLogNorm(1e-7,vmin=np.amin(counts),vmax=np.amax(counts)))#norm=SymLogNorm(0.01,vmin=-0.2,vmax=0.2))#cmap='inferno',vmin=0,vmax=0.02)
    
    zeropoint=25.9463
    for ap in [30]:
      aperture = CircularAperture([95,95], r=ap)

      phot_table = aperture_photometry(counts, aperture)
      tot_counts=float(phot_table['aperture_sum'])
      mag=-2.5*np.log10(tot_counts)+zeropoint
      print('Photometric measurement: {:04.2f}'.format(mag))
    
      annulus_masks = aperture.to_mask(method='center')
      annulus_data = annulus_masks[0].multiply(counts)

      circle2=plt.Circle([95.5,95.5],ap, fill=False,color='k',linestyle='--')
      ax.add_artist(circle2)
    #plt.show()
   
 
    print('Quasar: Host subtraction fit: ',head['1PS_MAG'])
    head = fits.getheader('testQuasarSamples_noHost/mcmc_out_mock_JWST_'+title+'_residual.fits')
    print('Quasar: Quasar subtraction fit: ',head['1PS_MAG'])


    indices={'SDSS':9,'MMBH':106,'CO':251,'WFIRST':684}
    if title=='MMBH':
      print('Database not saved for MMBHs')
    else:
      ##Direct from BlueTides:
      df=pd.read_pickle('/home/mmarshal/BLUETIDES/BlueTides/PIG_208/processed_data/quasarDatabase.pkl')
      print(len(df.loc[df['Sample']==title]))
      print(df.loc[(df['Sample']==title)&(df['Index']==indices[title])])
    
