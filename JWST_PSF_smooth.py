import numpy as np
import matplotlib
#matplotlib.use('Agg')
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
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def create_circular_mask(h, w, center=None, radius=None):
    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask


def plot_PSF(LPSF,axes,ivm_axes,f,err):
  print(LPSF)
  imgs = {}

  super_samp=2
  img = images.observed(f, cosmo, z, target_width_arcsec=width,smoothing = False,  verbose=True, PSF =PSFs[f],super_sampling=super_samp).particle(np.array([0.]), np.array([0.]), np.array([LPSF]),centre=[0,0,0])

  # create background image object (cutoutwidth in pixels)
  background_object = make_background.Background(zeropoint=zeropoint, pixel_scale=pixel_scale/super_samp, aperture_f_limit=aperture_f_limit, aperture_significance=ap_sig, aperture_radius=aperture_radius, verbose = True)

  mask=create_circular_mask(Npixels*super_samp, Npixels*super_samp, center=None, radius=np.floor(Npixels*super_samp/2))

  img_data=img.super.data*nJy_to_es 

  data_sigma=np.sqrt(np.abs(img_data)) #sigma = sqrt abs(signal)
  mx=np.amax(img_data)
  
  y,x=np.mgrid[0:len(img_data),0:len(img_data)]
  gauss=Gaussian2D(np.max(img_data)/5000,len(img_data)/2-1.5,len(img_data)/2-1.5,1.5,1.5)(x,y)
  ivm=1/((np.ones_like(gauss)*2e-6)+gauss)
  var=1/ivm
  mx_var=np.amax(ivm)
 
  img1 = axes.imshow(mask*(img_data),cmap='magma', norm=LogNorm(vmin = mx/10000, vmax=mx))
  img2 = err_axes.imshow(mask*ivm,cmap='magma', norm=LogNorm(vmin = mx_var/1e5, vmax=mx_var/10))#, vmin = mx -3, vmax=mx)
 
  
  hdu=fits.PrimaryHDU(mask*(img_data))
  
  if flux_fact!=20:
    hdu.writeto('data/sci_PSF_JWST_{}_{}'.format(filt_str,int(flux_fact))+'.fits',overwrite=True) 
  elif err>0.:
    hdu.writeto('data/sci_PSF_JWST_{}_{}'.format(filt_str,err).replace('.','p')+'.fits',overwrite=True)
  else:
    hdu.writeto('data/sci_PSF_JWST_{}_smooth.fits'.format(filt_str),overwrite=True)
  
  hdu_ivm=fits.PrimaryHDU(mask*ivm)
  #hdu_ivm=fits.PrimaryHDU(np.ones_like(data_sigma))
  #hdu_ivm=fits.PrimaryHDU(mask*1/((data_sigma)**2))
  if flux_fact!=20:
    hdu_ivm.writeto('data/ivm_PSF_JWST_{}_{}'.format(filt_str,int(flux_fact))+'.fits',overwrite=True) 
  elif err>0.:
    hdu_ivm.writeto('data/ivm_PSF_JWST_{}_{}'.format(filt_str,err).replace('.','p')+'.fits',overwrite=True)
  else:
    hdu_ivm.writeto('data/ivm_PSF_JWST_{}_smooth.fits'.format(filt_str),overwrite=True)

  axes.set_facecolor('black')

  cax1 = fig.add_axes([0.9, 0.11, 0.02, 0.77])
  cbar=fig.colorbar(img1, cax=cax1)#,norm=LogNorm(vmin=mx/1000,vmax=mx))#, ticks=[mx-3,mx-2,mx-1,mx])
  cbar.ax.set_ylabel(r'$\log(I/I_{\rm{max}})$')

  cax1 = err_fig.add_axes([0.9, 0.11, 0.02, 0.77])
  cbar=fig.colorbar(img2, cax=cax1)#, ticks=[mx-3,mx-2,mx-1,mx])
  cbar.ax.set_ylabel(r'$\log(I/I_{\rm{max}})$')
  return


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




if __name__=='__main__':
    #Setup
    cosmo = FLARE.default_cosmo()
    z = 6

    dust=True
    model = models.define_model('BPASSv2.2.1.binary/ModSalpeter_300') # DEFINE SED GRID -
    model.dust = {'A': 4.6, 'slope': -1.0}

    filters = [FLARE.filters.NIRCam_W[4]]
    filt_str=(filters[0].split('.')[-1])
    print('filter: ',filt_str)
    F = FLARE.filters.add_filters(filters, new_lam = model.lam* (1.+z))
    PSFs = PSF.Webb(filters, resampling_factor = 5) # creates a dictionary of instances of the webbPSF class

    width=3  #8.33 #size of cutout in ''  #MUST RESULT IN EVEN NUMBER OF PIXELS
    FOV=width/cosmo.arcsec_per_kpc_proper(z).value #size of cutout in kpc
    smoothing = None#('adaptive',60)
   
    pixel_scale = FLARE.filters.pixel_scale[filters[0]]     # arcsec/pixel (for NIRCam SW)
    Npixels = int(width/pixel_scale) #20#width of image / resolution
    
    #resolution = 0.065/cosmo.arcsec_per_kpc_proper(z).value #for WFC3 IR
    #.13 arcsec/pixel resolution for WFC3 IR
    #resolution = 0.13/cosmo.arcsec_per_kpc_proper(z).value
    
    resolution = 0.0125 #kpc/image pixel I think..., something to do with BT
    Ndim = int(FOV/resolution) #20#width of image / resolution
    #background setup
    aperture_radius = 2.5*pixel_scale         # aperture radius in arcsec
    zeropoint = 25.946              
    nJy_to_es = 1E-9 * 10**(0.4*(zeropoint-8.9))
    exp_time = 10000
    aperture_flux_limits={'JWST.NIRCAM.F090W':15.3, 'JWST.NIRCAM.F115W':13.2,
       'JWST.NIRCAM.F150W':10.6, 'JWST.NIRCAM.F200W':9.1, 'JWST.NIRCAM.F277W':14.3, 
       'JWST.NIRCAM.F356W':12.1, 'JWST.NIRCAM.F444W':23.6} #sensitivity at 10ks in nJy, 10 sigma
    aperture_f_limit = aperture_flux_limits[filters[0]]
    ap_sig = 10
    #https://jwst-docs.stsci.edu/near-infrared-camera/nircam-predicted-performance/nircam-sensitivity
    r = aperture_radius/pixel_scale # aperture radius in pixels

    if len(sys.argv)>1:
      flux_fact = np.float(sys.argv[1])
    else:
      flux_fact = 20
    print('flux_fact: ',flux_fact)

    if len(sys.argv)>2:
      err = np.float(sys.argv[2])
    else:
      err=0
    print('error: ',err)

    model.create_Fnu_grid(F, z, cosmo)

    folder='/home/mmarshal/BLUETIDES/BlueTides/PIG_208/processed_data/'

    fig, axes = plt.subplots(1,1, figsize = (5,5))
    err_fig, err_axes = plt.subplots(1,1, figsize = (5,5))

    #FPSF=13578*12 #12x BT model SDSS quasar brightness, from difference between PSF and quasar for SDSS-J0203
    #FPSF=16941.5654116655 * flux_fact
    FPSF=6e5#19489.507309311983 * flux_fact
    plot_PSF(FPSF,axes,err_axes,filters[0],err)
 
    #plt.savefig('/home/mmarshal/results/plots/BTpsfMC/mock_HST.pdf')
    plt.show()

