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


def plot_host(data,axes,f,dust=False):
  imgs = {}

  if dust:
    Fnu =  models.generate_Fnu_array(model, data.Masses, data.Ages, data.Metallicities, (10**model.dust['A'])*data.MetSurfaceDensities, F, f) # arrays of star particle fluxes in nJy
  else:
    Fnu = models.generate_Fnu_array(model, data.Masses, data.Ages, data.Metallicities, np.zeros_like(data.Masses), F, f) # arrays of star particle fluxes in nJy

  img = images.observed(f, cosmo, z, target_width_arcsec=width, smoothing = smoothing,  verbose=True, PSF =PSFs[f]).particle(data.X, data.Y, Fnu,centre=[0,0,0])
  mx=1.8

  # create background image object (cutoutwidth in pixels)
  background_object = make_background.Background(zeropoint=zeropoint, pixel_scale=pixel_scale, aperture_f_limit=aperture_f_limit, aperture_significance=ap_sig, aperture_radius=aperture_radius, verbose = True)
  img_bkg = background_object.create_background_image(Npixels)

  axes.imshow(np.log10((img.img.data+img_bkg.bkg)*nJy_to_es),cmap='magma', vmin = mx/1000, vmax=mx)
  axes.set_facecolor('black')
  return


def plot_host_quasar(data,Lquasar,axes,err_axes,f,dust=False,title=None):
  #Lquasar*=4
  print('Fquasar (nJy) ',Lquasar)
  imgs = {}

  if dust:
    Fnu = models.generate_Fnu_array(model, data.Masses, data.Ages, data.Metallicities, (10**model.dust['A'])*data.MetSurfaceDensities, F, f)  # arrays of star particle fluxes in nJy
  else:
    Fnu = models.generate_Fnu_array(model, data.Masses, data.Ages, data.Metallicities, np.zeros_like(data.Masses), F, f) # arrays of star particle fluxes in nJy

  #img_host = images.observed(f, cosmo, z, target_width_arcsec=width,smoothing = smoothing,  verbose=True, PSF =PSFs[f],super_sampling = 4).particle(np.append(data.X,0), np.append(data.Y,0), np.append(Fnu,Lquasar),centre=[0,0,0])
  super_samp=2
  #img_host= images.observed(f, cosmo, z, target_width_arcsec=width,smoothing = False,  verbose=True, PSF =PSFs[f],super_sampling=super_samp).particle(np.array([0.5]), np.array([0.5]), np.array([Lquasar]),centre=[0,0,0])
  img_host= images.observed(f, cosmo, z, target_width_arcsec=width,smoothing = False,  verbose=True, PSF =PSFs[f],\
super_sampling=super_samp).particle(np.array([0.]), np.array([0.]), np.array([Lquasar]),centre=[0,0,0])
  

  # create background image object (cutoutwidth in pixels)
  background_object = make_background.Background(zeropoint=zeropoint, pixel_scale=pixel_scale/super_samp, aperture_f_limit=aperture_f_limit, aperture_significance=ap_sig, aperture_radius=aperture_radius, verbose = True)
  img_bkg = background_object.create_background_image(Npixels*super_samp)


  img_bkg_data=img_bkg.bkg*nJy_to_es
  bkg_sigma=background_object.pixel.noise_es*np.ones_like(img_bkg.bkg)

  img_data=img_host.super.data*nJy_to_es 
  #img_data=img_host.img.data*nJy_to_es 
  
  y,x=np.mgrid[0:len(img_bkg.bkg),0:len(img_bkg.bkg)]
  gauss=Gaussian2D(np.max(img_data)/5000,34.5,34.5,2,2)(x,y)
  ivm=1/((bkg_sigma*super_samp)**2+(gauss))
  
  #data_sigma=np.sqrt(img_data)
  #ivm=1/((bkg_sigma*super_samp)**2+data_sigma**2)

  #ivm=np.ones_like(bkg_sigma)
  var=1/ivm

  
  mx=np.amax(img_data)
  #mx_var=np.amax(ivm)
  mx_var=np.amax(ivm)
  min_var=np.amin(ivm)
 
  img1 = axes.imshow(img_data+img_bkg_data,cmap='magma', norm=LogNorm(vmin = mx/10000, vmax=mx))
  #img2 = err_axes[0]rimshow(((bkg_sigma*super_samp)**2),cmap='magma', norm=LogNorm(vmin = min_var, vmax=mx_var))#, vmin = mx -3, vmax=mx)
  #img2 = err_axes[1].imshow((data_sigma**2),cmap='magma', norm=LogNorm(vmin = min_var, vmax=mx_var))#, vmin = mx -3, vmax=mx)
  
  #img2 = err_axes[1].imshow(gauss,cmap='magma', norm=LogNorm(vmin = min_var, vmax=mx_var))#, vmin = mx -3, vmax=mx)
  img2 = err_axes.imshow(ivm,cmap='magma', norm=LogNorm(vmin = min_var, vmax=mx_var))#, vmin = mx -3, vmax=mx)
 
  hdu=fits.PrimaryHDU(img_data+img_bkg_data)
  #err=0
  #uncertainty=1+err*np.random.standard_normal(np.shape(img_data))
  #data_sigma=data_sigma*uncertainty
  hdu_ivm=fits.PrimaryHDU(ivm)
  #hdu_ivm=fits.PrimaryHDU(np.ones_like(data_sigma))

  if onlyHost:
    if title:
      hdu.writeto('data/sci_mock_HST_{}_onlyHost.fits'.format(title),overwrite=True)
      hdu_ivm.writeto('data/ivm_mock_HST_{}_onlyHost.fits'.format(title),overwrite=True)
    else:
      hdu.writeto('data/sci_mock_HST_{}_onlyHost.fits',overwrite=True)
      hdu_ivm.writeto('data/ivm_mock_HST_{}_onlyHost.fits',overwrite=True)
  elif host:
    if title:
      hdu.writeto('data/sci_mock_HST_{}_host.fits'.format(title),overwrite=True)
      hdu_ivm.writeto('data/ivm_mock_HST_{}_host.fits'.format(title),overwrite=True)
    else:
      hdu.writeto('data/sci_mock_HST_host.fits',overwrite=True)
      hdu_ivm.writeto('data/ivm_mock_HST_host.fits',overwrite=True)
  else:
    if title:
      hdu.writeto('data/sci_mock_HST_{}.fits'.format(title),overwrite=True)
      hdu_ivm.writeto('data/ivm_mock_HST_{}.fits'.format(title),overwrite=True)
    else:
      hdu.writeto('data/sci_mock_HST.fits',overwrite=True)
      hdu_ivm.writeto('data/ivm_mock_HST.fits',overwrite=True)
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
    z = 6

    dust=True
    model = models.define_model('BPASSv2.2.1.binary/ModSalpeter_300') # DEFINE SED GRID -
    model.dust = {'A': 4.6, 'slope': -1.0}

    #filters = [FLARE.filters.NIRCam_W[4]]
    #F = FLARE.filters.add_filters(filters, new_lam = model.lam* (1.+z))
    #PSFs = PSF.Webb(filters, resampling_factor = 5) # creates a dictionary of instances of the webbPSF class
    filters = [FLARE.filters.WFC3NIR_W[3]]    
    F = FLARE.filters.add_filters(filters, new_lam = model.lam* (1.+z))
    PSFs = PSF.Hubble(filters)

    width=4.6#8.33#7.81 #size of cutout in ''  #MUST RESULT IN EVEN NUMBER OF PIXELS
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
    zeropoint = 25.946              # AB mag zeropoint, doesn't have any effect
    nJy_to_es = 1E-9 * 10**(0.4*(zeropoint-8.9))
    #aperture_f_limit = 9.1        # aperture flux limit (nJy) (F115W in 10ks, 10 sigma)
    aperture_f_limit = 48 #guess based on HST ivm maps        # aperture flux limit (nJy) (F115W in 10ks, 10 sigma)
    ap_sig = 10
    #https://jwst-docs.stsci.edu/near-infrared-camera/nircam-predicted-performance/nircam-sensitivity
    r = aperture_radius/pixel_scale # aperture radius in pixels

    #Quasar sample setup
    ###NOTE: Need to extract these numbers from BH_spectra_z7_dust
    #BHsamples=['SDSS_AGN_dust/9'] 
    #tau_UV=[1.165] #Min tau UV for MMBH, CO, WFIRST
    #dust_atten=np.exp(-np.array(tau_UV))#Need metallicity factor
    BHsamples=['MMBHs/106','SDSS_AGN_dust/9','CO_AGN_dust/251','WFIRST_AGN_dust/684'] 
    titles=['MMBH','SDSS','CO','WFIRST']
    tau_UV=[0.18,1.165,0.452,0.857] #Min tau UV for MMBH, CO, WFIRST
    dust_atten=np.exp(-np.array(tau_UV))#Need metallicity factor
    
    host=False
    onlyHost=False

    orientation='face_on'#None,'face_on','edge_on'    # Initialise background(s)


    model.create_Fnu_grid(F, z, cosmo)

    folder='/home/mmarshal/BLUETIDES/BlueTides/PIG_208/processed_data/'

    fig, axes = plt.subplots(1,1, figsize = (5,5))
    err_fig, err_axes = plt.subplots(1,1, figsize = (5,5))

    for ii,BH in enumerate(BHsamples):
      data = SynthObs.bluetides_data('PIG_208/processed_data/'+str(BH),dust=True)
      data=get_positions(data,orientation)
      Fquasar=load_quasar(folder+str(BH)+'/run_cloudy.con',filters[0],F)
      plot_host_quasar(data,Fquasar*dust_atten[ii],axes,err_axes,filters[0],dust=True,title=titles[ii])
 
    #plt.savefig('/home/mmarshal/results/plots/BTpsfMC/mock_HST.pdf')
    plt.show()

