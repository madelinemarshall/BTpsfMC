##Works with older version of SynthObs, dust model has been changed 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import SynthObs
import SynthObs.Morph
from SynthObs.SED import models
from SynthObs.Morph import measure
from SynthObs.Morph import images
from SynthObs.Morph import PSF
import FLARE.filters
from matplotlib.patches import Circle
import pandas as pd
from synphot import etau_madau
from mpl_toolkits.axes_grid1 import make_axes_locatable
import make_background
from photutils import aperture_photometry
from photutils import CircularAperture
from matplotlib.colors import LogNorm
from astropy.io import fits
from astropy.modeling.functional_models import Gaussian2D

matplotlib.rcParams['font.size'] = (9)
matplotlib.rcParams['figure.figsize'] = (7.3,7.3)
#plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def flux_host_quasar(data,Lquasar,f,ii,dust=False):
  #Lquasar*=4
  #print('Fquasar (nJy) ',Lquasar)

  imgs = {}

  if dust:
    Fnu = models.generate_Fnu_array(model, data.Masses, data.Ages, data.Metallicities, (10**model.dust['A'])*data.MetSurfaceDensities, F, f)  # arrays of star particle fluxes in nJy
  else:
    Fnu = models.generate_Fnu_array(model, data.Masses, data.Ages, data.Metallicities, np.zeros_like(data.Masses), F, f) # arrays of star particle fluxes in nJy

  return np.sum(Fnu)/Lquasar, np.sum(Fnu),Lquasar


def load_quasar(filename,f,F):
    prop='trans_reflect'
    dt=pd.read_csv(filename,sep='\t',header=0,names=['lambda','inc_cont','trans_cont','emit_cont',\
  'net_cont','refl_cont','trans_reflect','refl_line','out_line','line_label','cont_label','num_lines'])
    #dt=dt[(dt['lambda']>1e-3)&(dt['lambda']<1e1)]
    quasar_lambda=(dt['lambda']) #microns
    quasar_nuLnu=(dt[prop])
    quasar_nu=3e8/(quasar_lambda*1e-6)

    ##Need to mock Lyman-forest extinction on the redshifted galaxy spectra
    wave=np.array(dt['lambda']*1e4*(1+z)) #angstrom
    extcurve = etau_madau(wave, z)
    extinct_q=extcurve(wave)
    extinct_q[wave<10**3.5]=0 #This has an upturn at low-lambda - manually get rid of this

    quasar_Lnu=np.interp(np.log10(F[f].lam),np.flip(np.log10(wave),axis=0),np.flip(np.log10(quasar_nuLnu/quasar_nu*extinct_q),axis=0))
    quasar_Lnu[np.isnan(quasar_Lnu)]=0
    luminosity_distance = cosmo.luminosity_distance(z).to('cm').value
    measured_F=1E23 * 1E9 * np.trapz(np.multiply(10**quasar_Lnu,F[f].T), x = F[f].lam)/np.trapz(F[f].T, x = F[f].lam)* (1.+z) / (4. * np.pi * luminosity_distance**2)
    return measured_F


def get_positions(data,orientation=None):
  centre=data.BHposition
  if orientation=='edge_on':
    data.X, data.Y= edge_on(np.transpose([data.X, data.Y, data.Z]),data.Masses,th=resolution*Ndim,centre=centre)
    #data.X, data.Y= edge_on(np.transpose([data.X, data.Y, data.Z]),data.Masses,th=-1,centre=centre)
    centre=[0,0,0]
  elif orientation=='face_on':
    data.X, data.Y= face_on(np.transpose([data.X, data.Y, data.Z]),data.Masses,th=resolution*Ndim,centre=centre)
    #data.X, data.Y= face_on(np.transpose([data.X, data.Y, data.Z]),data.Masses,th=-1,centre=centre)
    centre=[0,0,0]
  else:
    print("ERROR: not sure what the centre will be - need to fix code")
  if len(data.X)==0:
    return

  data.X/=(1+z)
  data.Y/=(1+z)
  data.Z/=(1+z)
  return data


def inertia(pos,mass):  # pos in [[x1,x2..],[y1,y2..],[z1,z2..]]
    g11=np.sum((pos[1]*pos[1]+pos[2]*pos[2])*mass)
    g22=np.sum((pos[0]*pos[0]+pos[2]*pos[2])*mass)
    g33=np.sum((pos[0]*pos[0]+pos[1]*pos[1])*mass)
    g12=-np.sum(pos[0]*pos[1]*mass)
    g13=-np.sum(pos[0]*pos[2]*mass)
    g23=-np.sum(pos[1]*pos[2]*mass)
    g21=g12
    g31=g13
    g32=g23
    mx = np.array([[g11,g12,g13],[g21,g22,g23],[g31,g32,g33]])/np.sum(mass)
    w, v = np.linalg.eig(mx)
    v = v[:, np.abs(w).argsort()] # column v[:,i] is the eigenvector corresponding to the eigenvalue w[i]
    return v                      # return Imom eigenvector with eigenvalue from smallest to largest


def face_on(p,mass,th=-1,centre=None): # p in [[x1,y1,z1],[x2,y2,z2],...]
    if centre.all():
      pass
    else:
      centre = np.average(p,axis=0)
    dr = p - centre
    rr = np.linalg.norm(dr,axis=1)
    if th>0:
        mask = rr<th
    else:
        mask = rr>0
    if len(dr[mask])==0:
      return [],[]
    v = inertia(np.transpose(dr[mask]),mass[mask])
    xnew = np.einsum('ij,j->i', dr, v[:,1])
    ynew = np.einsum('ij,j->i', dr, v[:,0])
    return xnew,ynew


if __name__=='__main__':
  #Setup
  cosmo = FLARE.default_cosmo()
  z = 7

  fig,ax=plt.subplots()
  dust=True
  model = models.define_model('BPASSv2.2.1.binary/ModSalpeter_300') # DEFINE SED GRID -
  model.dust = {'A': 4.6, 'slope': -1.0}

  filter_set = FLARE.filters.NIRCam_W[2:]
  filter_set.append(FLARE.filters.MIRI[0])
  filter_set.append(FLARE.filters.MIRI[1])
  
  folder='/home/mmarshal/BLUETIDES/BlueTides/PIG_208/processed_data/'

  df=pd.read_pickle('/home/mmarshal/BLUETIDES/BlueTides/PIG_208/processed_data/quasarDatabase.pkl')
  #Quasar sample setup
  sample='SDSS'
  num_samps=len(df.loc[(df['Sample']==sample)])
    
  flux_ratio = np.zeros((len(filter_set),num_samps))
  host_flux = np.zeros((len(filter_set),num_samps))
  AGN_flux = np.zeros((len(filter_set),num_samps))

  for jj,filters in enumerate(filter_set):
    print(filters)
    filt_str=(filters.split('.')[-1])
    filters=[filters]
    #print('filter: ',filt_str)
    F = FLARE.filters.add_filters(filters, new_lam = model.lam* (1.+z))
    PSFs = PSF.Webb(filters, resampling_factor = 5) # creates a dictionary of instances of the webbPSF class

    width=3 #size of cutout in ''
    FOV=width/cosmo.arcsec_per_kpc_proper(z).value #size of cutout in kpc
    smoothing = None#('adaptive',60)
   
    pixel_scale = FLARE.filters.pixel_scale[filters[0]]     # arcsec/pixel (for NIRCam SW)
    Npixels = int(width/pixel_scale) #20#width of image / resolution
    
    resolution = 0.0125 #kpc/image pixel I think...
    Ndim = int(FOV/resolution) #20#width of image / resolution


    orientation='face_on'#None,'face_on','edge_on'    # Initialise background(s)


    model.create_Fnu_grid(F, z, cosmo)

    
    for ii in range(0,num_samps):
    
      index=df.loc[(df['Sample']==sample)].iloc[ii]['Index']
      if index!=41:
        tau_UV=df.loc[(df['Sample']==sample)].iloc[ii]['tau_UV_AGN']
        dust_atten=np.exp(tau_UV)

        BH=sample+'_AGN_dust/'+str(index)


        data = SynthObs.bluetides_data('PIG_208/processed_data/'+str(BH),dust=True)
        data=get_positions(data,orientation)
        Fquasar=load_quasar(folder+str(BH)+'/run_cloudy.con',filters[0],F)
        flux_ratio[jj,ii],host_flux[jj,ii],AGN_flux[jj,ii]=flux_host_quasar(data,Fquasar*dust_atten,filters[0],ii,dust=True)

    plt.hist(np.log10(flux_ratio[flux_ratio>0]),histtype='step',label=filt_str,range=(-2.4,0))
  plt.legend()
  plt.show()
  np.save('flux_ratio.npy',flux_ratio)
  np.save('host_flux.npy',host_flux)
  np.save('AGN_flux.npy',AGN_flux)
