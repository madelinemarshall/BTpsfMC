import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from photutils import CircularAperture,CircularAnnulus
from photutils import aperture_photometry, background
import matplotlib
from matplotlib.colors import SymLogNorm
import sys
import pandas as pd
from scipy.ndimage import gaussian_filter
from astropy.visualization import AsinhStretch, LogStretch, SinhStretch, SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.convolution import Gaussian2DKernel, Tophat2DKernel, Box2DKernel, interpolate_replace_nans
from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid
from scipy.interpolate import interp1d


matplotlib.rcParams['font.size'] = (8)
#plt.rc('text', usetex=True)
plt.rc('font', family='serif')
markers= ['s','^','o','d','h','p']
_stretch = AsinhStretch()
_stretch.a = (0.01 - 0.0001)/2 / (0.01+0.0001)

def flux_to_mag(flux):
    return -2.5*np.log10(flux/(3631e9)) #flux in nJy

def full_mag(counts):
    return -2.5*np.log10(np.sum(counts))+25.9463

def sersic_rad(filt,ii):
    if filt=='F200W':
      _pattern = 'runJWST/SDSS_z7_SN/mcmc_out_mock_JWST_SDSS_{}_residual.fits'
      head = fits.getheader(_pattern.format(ii))
    else:
      _pattern = 'runJWST/SDSS_z7_{}/mcmc_out_mock_JWST_SDSS_{}_residual.fits'
      head = fits.getheader(_pattern.format(filt,ii))
    fit_rad=head['2SER_RE'].split(' ')[0]
    err_rad=head['2SER_RE'].split(' ')[-1]
    #fit_sers[jj]=head['2SER_N'].split(' ')[0]
    #err_sers[jj]=head['2SER_N'].split(' ')[-1]
    return fit_rad,err_rad

                
def sersic_rad_true(filt,ii):
    if filt=='F200W':
      _pattern_OH= 'runJWST/SDSS_z7_SN_onlyHost/mcmc_out_mock_JWST_SDSS_{}_residual.fits'
    else:
      _pattern_OH= 'runJWST/SDSS_z7_{}_onlyHost/mcmc_out_mock_JWST_SDSS_{}_residual.fits'
    head_OH = fits.getheader(_pattern_OH.format(filt,ii))
    OH_rad=head_OH['1SER_RE'].split(' ')[0]
    OH_rad_err=head_OH['1SER_RE'].split(' ')[-1]
    return OH_rad,OH_rad_err
      

def ext_dict(dd):
    return np.concatenate(list(dd.values()))


def find_eff_half_light_radius(pix_area,z):
    #Find the area (non-contiguous) which contains 50% of the luminosity
    ordered_l = np.sort(z,axis=None)[::-1]
    total_l=np.sum(ordered_l)
    sum_l = np.cumsum(ordered_l)
    num_above_half=(np.size(ordered_l[sum_l<total_l/2]))+1 #number of pixels containining 50% of luminosity
    #lum_half=ordered_l[sum_l<total_l/2][-1]   

    area = pix_area * num_above_half
    radius = np.sqrt(area/np.pi)
    return radius #, (lum_half,sum_l[sum_l<total_l/2][-1],total_l/2)


def find_half_light_radius(x,y,z):
    centre = np.unravel_index(np.argmax(z), np.array(z).shape) #Peak flux location
    total_l=np.sum(z)

    rad = np.linspace(5,np.shape(z)[0]-15,np.shape(z)[0])
    counts_in_r=np.zeros(len(rad))
    for ii,rr in enumerate(rad): 
      aperture = CircularAperture(centre, r=rr)

      #phot_table = aperture_photometry(counts_masked-sky, aperture)
      phot_table = aperture_photometry(z, aperture)
      counts_in_r[ii]=float(phot_table['aperture_sum'])

    #interpolate to find sub-pixel half-radius        
    f = interp1d(rad,counts_in_r, kind='cubic')
    xnew = np.linspace(np.min(rad), np.max(rad), num=80, endpoint=True)
    half_interp = xnew[np.argmax(f(xnew)>total_l/2)]
    """
    plt.figure()
    plt.plot(rad,counts_in_r,'o')
    plt.plot(xnew, f(xnew))
    plt.axvline(half_interp,color='r')
    plt.axhline(total_l/2)
    plt.show()
    """   
    return half_interp

def masking(counts,kernel):
    cent=[int(len(counts)/2)-1,int(len(counts)/2)-1]
    crop_cent=[int(len(counts)/2)-2,int(len(counts)/2)-2]
    crop_rad=5

     
    inner_ap = CircularAperture(crop_cent, r=crop_rad)
    inner_mask = inner_ap.to_mask(method='center').to_image(np.shape(counts))
    #inner_mask_shape = inner_mask.to_image((192,192))
    counts[inner_mask>0]=np.nan
    if kernel:
        # create a "fixed" image with NaNs replaced by interpolated values
        counts = interpolate_replace_nans(counts, kernel)
    else:
        ring_ap = CircularAnnulus(crop_cent, r_in=crop_rad,r_out=crop_rad+1)
        ring_dat=ring_ap.to_mask(method='center').multiply(counts)
        counts[np.isnan(counts)]=np.amin(ring_dat[ring_dat>0])

    return counts


if __name__=='__main__':
  indices=np.array([2,3,6,7,8,9,10,12,16,18,20,22,23,25,27,32,36,40,41,43,45,46,100])

  filt_no={'F115W':0,'F150W':1,'F200W':2,'F277W':3,'F356W':4,'F444W':5,'F560W':6,'F770W':7}
  wavelength={'F115W':1.15,'F150W':1.50,'F200W':2.00,'F277W':2.77,'F356W':3.56,'F444W':4.44}

  df=pd.read_pickle('/home/mmarshal/BLUETIDES/BlueTides/PIG_208/processed_data/quasarDatabase_processed.pkl')
  df=df[df['Sample']=='SDSS']

  half_rad={}
  half_rad_true={}
  eff_rad={}
  eff_rad_min={}
  eff_rad_gauss={}
  eff_rad_true={}
  rad_sersic={}
  err_sersic={}
  rad_sersic_true={}
  err_sersic_true={}

  #fig3,axes3=plt.subplots(2,1,figsize=(3.4,6),gridspec_kw={'hspace':0.05,'left':0.15},sharex=True)
  #fig3,axes3=plt.subplots(3,1,sharex=True,sharey=True,figsize=(3.4,8),gridspec_kw={'hspace':0.05,'bottom':0.07,'top':0.93})
  fig3,axes3=plt.subplots(3,1,sharex=True,sharey=True,figsize=(3.2,6),gridspec_kw={'hspace':0.05,'bottom':0.1,'top':0.9})
  fig,axes=plt.subplots(1,1,figsize=(3.2,2.1),gridspec_kw={'hspace':0.05,'left':0.265,'right':0.75,'top':0.96,'bottom':0.18})
  
  for ff,filt in enumerate(['F115W','F150W','F200W','F277W','F356W','F444W']):#,'F560W','F770W']):
  #for ff,filt in enumerate(['F277W']):#,'F356W','F444W']):#,'F560W','F770W']):
    half_rad[filt]=np.zeros(len(indices)-1) 
    half_rad_true[filt]=np.zeros(len(indices)-1) 
    eff_rad[filt]=np.zeros(len(indices)-1) 
    eff_rad_min[filt]=np.zeros(len(indices)-1) 
    eff_rad_gauss[filt]=np.zeros(len(indices)-1) 
    eff_rad_true[filt]=np.zeros(len(indices)-1) 
    rad_sersic[filt]=np.zeros(len(indices)-1) 
    err_sersic[filt]=np.zeros(len(indices)-1) 
    rad_sersic_true[filt]=np.zeros(len(indices)-1) 
    err_sersic_true[filt]=np.zeros(len(indices)-1) 
    
    print('______________________________'+filt+'________________________________')
    
    zeropoint=25.9463
    if filt in ['F277W','F356W','F444W']:
        pxscale = 0.063/2 #arcsec
        #pxscale = 1
    else:
        pxscale = 0.031/2
        #pxscale = 1
    
    FWHM=0.2516*wavelength[filt]/6.57/pxscale
    pxscale *= 0.269/0.05 #to kpc
    crop_rad = 5
    for ii,ind in enumerate(indices):
      if ind>41:
        ii-=1
      if ind!=41:
        #load images and flux values
        #ind=str(sys.argv[1])
        title='SDSS_'+str(ind)
        
        if filt=='F200W':
          counts = fits.getdata('runJWST/SDSS_z7_SN/mcmc_out_mock_JWST_'+title+'_point_source_subtracted.fits')
        else:
          counts = fits.getdata('runJWST/SDSS_z7_'+filt+'/mcmc_out_mock_JWST_'+title+'_point_source_subtracted.fits')
        counts_gauss = gaussian_filter(counts, (1, 1))
        #counts_min = np.copy(counts)
        #counts_gauss = np.copy(counts)
        #counts_min = masking(counts_min,False)
        #counts_gauss = masking(counts_gauss,kernel = Gaussian2DKernel(x_stddev=FWHM))
     
        x,y = np.meshgrid(np.arange(np.shape(counts)[0]), np.arange(np.shape(counts)[1]))
        
        true_counts =  fits.getdata('data/sci_mock_JWST_{}_SDSS_{}_onlyHost.fits'.format(filt,ind))
        true_counts = gaussian_filter(true_counts, (1, 1))
        
        half_rad[filt][ii]=(find_half_light_radius(x,y,counts))*pxscale
        eff_rad[filt][ii]=(find_eff_half_light_radius(1,counts))*pxscale
        #eff_rad_min[filt][ii]=(find_eff_half_light_radius(1,counts_min))*pxscale
        eff_rad_gauss[filt][ii]=(find_eff_half_light_radius(1,counts_gauss))*pxscale
        
        half_rad_true[filt][ii]=(find_half_light_radius(x,y,true_counts))*pxscale
        eff_rad_true[filt][ii]=(find_eff_half_light_radius(1,true_counts))*pxscale
    
        rad_sersic[filt][ii], err_sersic[filt][ii] = sersic_rad(filt,ind)
        #print(rad_sersic[filt][ii])
        rad_sersic[filt][ii]*=pxscale
        err_sersic[filt][ii]*=pxscale
        
        ##Runs don't exist
        #rad_sersic_true[filt][ii], err_sersic_true[filt][ii] = sersic_rad_true(filt,ind)
        #rad_sersic_true[filt][ii]*=pxscale
        #err_sersic_true[filt][ii]*=pxscale

    #print('Sersic')
    print('Sersic Radius')
    prop=rad_sersic[filt]-eff_rad_true[filt]
    print(np.percentile(prop[df[filt]==True],[5,50,95]))
    print(np.percentile(prop[df[filt]==True],[5,50,95])-np.median(prop[df[filt]==True]))
    print('Effective Radius')
    prop=eff_rad[filt]-eff_rad_true[filt]
    print(np.percentile(prop[df[filt]==True],[5,50,95]))
    print(np.percentile(prop[df[filt]==True],[5,50,95])-np.median(prop[df[filt]==True]))
    print('Effective Radius, Masked')
    prop=eff_rad_gauss[filt]-eff_rad_true[filt]
    print(np.percentile(prop[df[filt]==True],[5,50,95]))
    print(np.percentile(prop[df[filt]==True],[5,50,95])-np.median(prop[df[filt]==True]))
  
    #best_guess=eff_rad_gauss[filt]
    #up_err=eff_rad[filt]-best_guess
    #low_err=best_guess-eff_rad_min[filt]

    #Theoretical radius
    df_T = df[df[filt]==True]
    #axes.plot(df_T['Radius'],rad_sersic[filt][df[filt]==True],marker=markers[ff],label=filt,color='C'+str(ff),markerfacecolor='w',ls='',markeredgewidth=1.5)
    axes.errorbar(df_T['Radius'],rad_sersic[filt][df[filt]==True],yerr=err_sersic[filt][df[filt]==True],marker=markers[ff],color='C'+str(ff),markerfacecolor='w',ls='',label=filt,markeredgewidth=1.5)

    ##True effective vs sersic
    axes3[0].errorbar(eff_rad_true[filt][df[filt]==True],rad_sersic[filt][df[filt]==True],yerr=err_sersic[filt][df[filt]==True],marker=markers[ff],color='C'+str(ff),markerfacecolor='w',ls='',label=filt,markeredgewidth=1.5)

    ##True effective vs effective
    axes3[1].plot(eff_rad_true[filt][df[filt]==True],eff_rad[filt][df[filt]==True],marker=markers[ff],color='C'+str(ff),markerfacecolor='w',ls='',label=filt,markeredgewidth=1.5)
    #axes3[1].errorbar(rad_sersic_true[filt][df[filt]==True],rad_sersic[filt][df[filt]==True],xerr=err_sersic_true[filt][df[filt]==True],yerr=err_sersic[filt][df[filt]==True],marker=markers[ff],color='C'+str(ff),markerfacecolor='w',ls='',label=filt,markeredgewidth=1.5)

    ##True effective vs masked effective
    axes3[2].plot(eff_rad_true[filt][df[filt]==True],eff_rad_gauss[filt][df[filt]==True],marker=markers[ff],label=filt,color='C'+str(ff),markerfacecolor='w',ls='',markeredgewidth=1.5)
    #axes312].plot(half_rad_true[filt][df[filt]==True],half_rad[filt][df[filt]==True],marker=markers[ff],label=filt,color='C'+str(ff),markerfacecolor='w',ls='',markeredgewidth=1.5)
    #axes3[2].errorbar(eff_rad_true[filt][df[filt]==True],best_guess[df[filt]==True],yerr=[low_err[df[filt]==True],up_err[df[filt]==True]],marker=markers[ff],ls='',label=filt,color='C'+str(ff),markerfacecolor='w',markeredgewidth=1.5)

  #print(half_rad)
  #print(half_rad_true)
  #print(eff_rad)
  #print(eff_rad_true)

  for ff in range(0,3):
    axes3[ff].plot([0,3],[0,3],'k')
    #axes3[ff].set_yticks([26,25,24,23])
    axes3[ff].axis('square')
    axes3[ff].set_xlim(0.45,2.6)
    axes3[ff].set_ylim(0.45,2.6)
    #axes3[ff].set_yticks([0.1,0.2,0.3,0.4,0.5])
  axes.set_ylim(0.4,1.9)
  axes.set_xlim(0.24,0.85)
  axes.plot([0,3],[0,3],'k')
  axes.set_yticks([0.5,1.0,1.5])
  
  #axes3[0].text(24.4,25.85,'Median: {0:.3f}'.format(np.median(ext_dict(mag_sersic_fit)-ext_dict(mag_true))))
  #axes3[0].text(24.1,26.15,r'$\sigma$: {0:.3f}'.format(np.std(ext_dict(mag_sersic_fit)-ext_dict(mag_true))))

  axes.set_ylabel('Best-Fit\nSersic Radius (kpc)')
  axes.set_xlabel('Half-Mass Radius (kpc)')
  axes.set_aspect(1 / axes.get_data_ratio())
  axes3[0].set_ylabel('Best-Fit\nSersic Radius (kpc)')
  axes3[1].set_ylabel('Effective\nRadius (kpc)')
  axes3[2].set_ylabel('Masked\nEffective Radius (kpc)')
  axes3[2].set_xlabel('True Host\nEffective Radius (kpc)')
  axes3[0].legend(loc=(-0.45,1.05),ncol=3)
 
  """
  for ff in range(0,6):
     #ax2[ff].set_yscale('log')
     #ax2[ff].set_xscale('log')
    ax2[ff].plot([23,27],[23,27])
    ax2[ff].set_xlim(23.5,26.5)
    ax2[ff].set_ylim(23.5,26.5)
  """

#  plt.legend()
  #axes3[2].invert_xaxis()
  #axes3[2].invert_yaxis()
  fig.savefig('compare_radii_fits.pdf')
  fig3.savefig('compare_radii_blocking.pdf')
  plt.show()
