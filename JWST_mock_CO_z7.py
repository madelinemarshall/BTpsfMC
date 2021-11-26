import numpy as np
import matplotlib
matplotlib.use('Agg')
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


def plot_host_quasar(data,Lquasar,axes,err_axes,f,exp_time,dust=False,title=None):
  #Lquasar*=4
  #print('Fquasar (nJy) ',Lquasar)

  imgs = {}

  if dust:
    Fnu = models.generate_Fnu_array(model, data.Masses, data.Ages, data.Metallicities, (10**model.dust['A'])*data.MetSurfaceDensities, F, f)  # arrays of star particle fluxes in nJy
  else:
    Fnu = models.generate_Fnu_array(model, data.Masses, data.Ages, data.Metallicities, np.zeros_like(data.Masses), F, f) # arrays of star particle fluxes in nJy

  super_samp=2

  if onlyHost:
    img_host = images.observed(f, cosmo, z, target_width_arcsec=width,smoothing = False,  verbose=False, \
    PSF =PSFs[f],super_sampling = super_samp).particle(data.X, data.Y, Fnu,centre=[0,0,0])
  elif host:
    img_host = images.observed(f, cosmo, z, target_width_arcsec=width,smoothing = False,  verbose=False, \
    PSF =PSFs[f],super_sampling = super_samp).particle(np.append(data.X,0.), np.append(data.Y,0.), np.append(Fnu,Lquasar),centre=[0,0,0])
  else:
    img_host= images.observed(f, cosmo, z, target_width_arcsec=width,smoothing = False,  verbose=False, PSF =PSFs[f],\
    super_sampling=super_samp).particle(np.array([0.]), np.array([0.]), np.array([Lquasar]),centre=[0,0,0])
  
  img_data=img_host.super.data*nJy_to_es 
 
  #Add shot noise 
  full_img = img_data * exp_time
  full_img[full_img<0] = 0
  noisy_full_img = np.random.poisson(full_img)
  img_data = noisy_full_img / exp_time

  # create background image object (cutoutwidth in pixels)
  background_object = make_background.Background(zeropoint=zeropoint, pixel_scale=pixel_scale/super_samp, \
aperture_f_limit=aperture_f_limit, aperture_significance=ap_sig, aperture_radius=aperture_radius, verbose = False)
  img_bkg = background_object.create_background_image(Npixels*super_samp)


  img_bkg_data=img_bkg.bkg*nJy_to_es
  bkg_sigma=background_object.pixel.noise_es*np.ones_like(img_bkg.bkg)

  #data_sigma=np.sqrt(img_data)
  
  y,x=np.mgrid[0:len(img_bkg.bkg),0:len(img_bkg.bkg)]

  #if filt_str=='F444W':
  #  gauss=Gaussian2D(np.max(img_data)/5000,len(img_bkg.bkg)/2-1.5,len(img_bkg.bkg)/2-1.5,2,2)(x,y)
  #elif filt_str=='F356W':
  #  gauss=Gaussian2D(np.max(img_data)/5000,len(img_bkg.bkg)/2-1.5,len(img_bkg.bkg)/2-1.5,4,4)(x,y)
  #else:
  gauss=Gaussian2D(np.max(img_data)/5000,len(img_bkg.bkg)/2-1.5,len(img_bkg.bkg)/2-1.5,2,2)(x,y)
  #print('Center loc: ',len(img_bkg.bkg)/2-1.5) 
  ivm=1/((bkg_sigma*super_samp)**2+(gauss))
  var=1/ivm

  mx=np.amax(img_data)
  mx_var=np.amax(ivm)
 
  img1 = axes.imshow(img_data+img_bkg_data,cmap='magma', norm=LogNorm(vmin = mx/10000, vmax=mx))
  img2 = err_axes.imshow(ivm,cmap='magma', norm=LogNorm(vmin = mx_var/1e5, vmax=mx_var/10))#, vmin = mx -3, vmax=mx)
 
  hdu=fits.PrimaryHDU(img_data+img_bkg_data)
  hdu_ivm=fits.PrimaryHDU(ivm)
  #hdu_ivm=fits.PrimaryHDU(np.ones_like(data_sigma)*10000)
  
  if onlyHost:
    if title:
      if exp_time==10000:
        hdu.writeto('data/sci_mock_JWST_{}_{}_onlyHost.fits'.format(filt_str,title),overwrite=True)
        hdu_ivm.writeto('data/ivm_mock_JWST_{}_{}_onlyHost.fits'.format(filt_str,title),overwrite=True)
      else:
        hdu.writeto('data/sci_mock_JWST_{}_{}_onlyHost_{}s.fits'.format(filt_str,title,exp_time),overwrite=True)
        hdu_ivm.writeto('data/ivm_mock_JWST_{}_{}_onlyHost_{}s.fits'.format(filt_str,title,exp_time),overwrite=True)
    else:
      hdu.writeto('data/sci_mock_JWST_{}_onlyHost.fits'.format(filt_str),overwrite=True)
      hdu_ivm.writeto('data/ivm_mock_JWST_{}_onlyHost.fits'.format(filt_str),overwrite=True)
  elif host:
    if title:
      if exp_time==10000:
        hdu.writeto('data/sci_mock_JWST_{}_{}_host_SN.fits'.format(filt_str,title),overwrite=True)
        hdu_ivm.writeto('data/ivm_mock_JWST_{}_{}_host_SN.fits'.format(filt_str,title),overwrite=True)
      else:
        hdu.writeto('data/sci_mock_JWST_{}_{}_host_SN_{}s.fits'.format(filt_str,title,exp_time),overwrite=True)
        hdu_ivm.writeto('data/ivm_mock_JWST_{}_{}_host_SN_{}s.fits'.format(filt_str,title,exp_time),overwrite=True)
    else:
      hdu.writeto('data/sci_mock_JWST_{}_host.fits'.format(filt_str),overwrite=True)
      hdu_ivm.writeto('data/ivm_mock_JWST_{}_host.fits'.format(filt_str),overwrite=True)
  else:
    if title:
      hdu.writeto('data/sci_mock_JWST_{}_{}.fits'.format(filt_str,title),overwrite=True)
      hdu_ivm.writeto('data/ivm_mock_JWST_{}_{}.fits'.format(filt_str,title),overwrite=True)
    else:
      hdu.writeto('data/sci_mock_JWST_{}.fits'.format(filt_str),overwrite=True)
      hdu_ivm.writeto('data/ivm_mock_JWST_{}.fits'.format(filt_str),overwrite=True)

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
    z = 7

    dust=True
    model = models.define_model('BPASSv2.2.1.binary/ModSalpeter_300') # DEFINE SED GRID -
    model.dust = {'A': 4.6, 'slope': -1.0}

    #####HOST?
    host=True
    onlyHost=True
    if host and onlyHost:
      print('__________ONLY HOST___________')
    elif host:
      print('__________HOST___________')
    else:
      print('__________ONLY QUASAR___________')

    filters = [FLARE.filters.NIRCam_W[4]]
    #filters = [FLARE.filters.MIRI[0]]
    filt_str=(filters[0].split('.')[-1])
    print('filter: ',filt_str)
    F = FLARE.filters.add_filters(filters, new_lam = model.lam* (1.+z))
    PSFs = PSF.Webb(filters, resampling_factor = 5) # creates a dictionary of instances of the webbPSF class

    width=3 #size of cutout in ''
    FOV=width/cosmo.arcsec_per_kpc_proper(z).value #size of cutout in kpc
    smoothing = None#('adaptive',60)
   
    pixel_scale = FLARE.filters.pixel_scale[filters[0]]     # arcsec/pixel (for NIRCam SW)
    Npixels = int(width/pixel_scale) #20#width of image / resolution
    
    #resolution = 0.065/cosmo.arcsec_per_kpc_proper(z).value #for WFC3 IR
    #.13 arcsec/pixel resolution for WFC3 IR
    #resolution = 0.13/cosmo.arcsec_per_kpc_proper(z).value
    
    resolution = 0.0125 #kpc/image pixel I think...
    Ndim = int(FOV/resolution) #20#width of image / resolution
    #background setup
    aperture_radius = 2.5*pixel_scale         # aperture radius in arcsec
    zeropoint = 25.946              # AB mag zeropoint, doesn't have any effect
    nJy_to_es = 1E-9 * 10**(0.4*(zeropoint-8.9))
    exp_time = 10000
    if exp_time==10000:
      aperture_flux_limits={'JWST.NIRCAM.F090W':15.3, 'JWST.NIRCAM.F115W':13.2,
       'JWST.NIRCAM.F150W':10.6, 'JWST.NIRCAM.F200W':9.1, 'JWST.NIRCAM.F277W':14.3, 
       'JWST.NIRCAM.F356W':12.1, 'JWST.NIRCAM.F444W':23.6,'JWST.MIRI.F560W':130} #sensitivity at 10ks in nJy, 10 sigma
      aperture_f_limit = aperture_flux_limits[filters[0]]
    else:
      if filters[0]=='JWST.NIRCAM.F200W':
        aperture_flux_limits={1000:44.9,5000:13.1,10000:9.1} #10 sigma limits for F200W, 1ks, 5ks, 10ks
      elif filters[0]=='JWST.NIRCAM.F150W': 
        aperture_flux_limits={1000:52.2,4800:15.55,5000:15.2,10000:10.6} #10 sigma limits for F150W, 1ks, 5ks, 10ks
      else:
        print("ERR, can't have not specified filter with non-10ks exposure time")
      aperture_f_limit = aperture_flux_limits[exp_time]
    print('Aperture flux limit ',aperture_f_limit,'Exp time ',exp_time)
      

    ap_sig = 10
    #https://jwst-docs.stsci.edu/near-infrared-camera/nircam-predicted-performance/nircam-sensitivity
    r = aperture_radius/pixel_scale # aperture radius in pixels

    #Quasar sample setup
    sample='CO'

    orientation='face_on'#None,'face_on','edge_on'    # Initialise background(s)


    model.create_Fnu_grid(F, z, cosmo)

    folder='/home/mmarshal/BLUETIDES/BlueTides/PIG_208/processed_data/'

    df=pd.read_pickle('/home/mmarshal/BLUETIDES/BlueTides/PIG_208/processed_data/quasarDatabase.pkl')
    
    num_samps=len(df.loc[(df['Sample']==sample)])
    
    
       
    for ii in range(0,num_samps):
      fig, axes = plt.subplots(1,1, figsize = (5,5))
      err_fig, err_axes = plt.subplots(1,1, figsize = (5,5))
    
      index=df.loc[(df['Sample']==sample)].iloc[ii]['Index']
      print('INDEX {}'.format(index))
      if (index!=96) and (index!=336):
        tau_UV=df.loc[(df['Sample']==sample)].iloc[ii]['tau_UV_AGN']
        dust_atten=np.exp(tau_UV)

        BH=sample+'_AGN_dust/'+str(index)
        #print(BH,tau_UV)


        data = SynthObs.bluetides_data('PIG_208/processed_data/'+str(BH),dust=True)
        data=get_positions(data,orientation)
        Fquasar=load_quasar(folder+str(BH)+'/run_cloudy.con',filters[0],F)
        plot_host_quasar(data,Fquasar*dust_atten,axes,err_axes,filters[0],exp_time,dust=True,title=sample+'_'+str(index))
      plt.close()
 
    #plt.savefig('/home/mmarshal/results/plots/BTpsfMC/mock_F200W.pdf')
    #plt.show()

