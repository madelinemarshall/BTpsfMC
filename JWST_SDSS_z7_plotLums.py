##Works with older version of SynthObs, dust model has been changed 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
import sys
import os
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

matplotlib.rcParams['font.size'] = (9)
matplotlib.rcParams['figure.figsize'] = (7.3,7.3)
#plt.rc('text', usetex=True)
plt.rc('font', family='serif')

wavelengths=[1.15,1.50,2.00,2.77,3.56,4.44,5.60,7.70]

flux_ratio=np.load('flux_ratio.npy')
host_flux=np.load('host_flux.npy')
AGN_flux=np.load('AGN_flux.npy')

df=pd.read_pickle('/home/mmarshal/BLUETIDES/BlueTides/PIG_208/processed_data/quasarDatabase_processed.pkl')
df=df[df['Sample']=='SDSS']

detectability=np.zeros(len(df['F115W'])) #detectable in how many filters?   
for filt in ['F115W','F150W','F200W','F277W','F356W','F444W']:
      detectability+=np.array(df[filt],dtype=int)
easy=detectability>4


"""
fig,ax=plt.subplots()
for jj in range(0,np.shape(flux_ratio)[0]):
  plt.hist(np.log10(flux_ratio[jj,flux_ratio[jj]>0]),histtype='step',range=(-2.4,0))
"""
print(easy)
fig,ax=plt.subplots()
for ii in range(0,np.shape(flux_ratio)[1]):
  if ii<18:
    if easy[ii]==1:
      plt.plot(wavelengths,np.log10(flux_ratio[:,ii]),'-')
    else:
      plt.plot(wavelengths,np.log10(flux_ratio[:,ii]),'r:')
  if ii>18:
    if easy[ii-1]==1:
      plt.plot(wavelengths,np.log10(flux_ratio[:,ii]),'-')
    else:
      plt.plot(wavelengths,np.log10(flux_ratio[:,ii]),'r:')
plt.plot(wavelengths,np.log10(np.median(flux_ratio,axis=1)),'k')


fig,ax=plt.subplots()
plt.plot(wavelengths,np.log10(np.median(flux_ratio,axis=1)),'o-',label='Flux Ratio')
plt.plot(wavelengths,np.log10(np.median(host_flux,axis=1)),'s-',label='Host Flux')
plt.plot(wavelengths,np.log10(np.median(AGN_flux,axis=1)),'^-',label='AGN Flux')
#plt.plot(wavelengths,((np.median(flux_ratio,axis=1))-(np.median(flux_ratio,axis=1)[0]))/(np.median(flux_ratio,axis=1)[0]),'o-',label='Flux Ratio')
#plt.plot(wavelengths,((np.median(host_flux,axis=1))-(np.median(host_flux,axis=1)[0]))/(np.median(host_flux,axis=1)[0]),'s-',label='Host Flux')
#plt.plot(wavelengths,((np.median(AGN_flux,axis=1))-(np.median(AGN_flux,axis=1)[0]))/(np.median(AGN_flux,axis=1)[0]),'^-',label='AGN Flux')
plt.ylabel('Relative to 1.15 micron')
plt.xlabel('Wavelength (micron)')
plt.legend()
plt.show()
