import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import SynthObs
from SynthObs.SED import models
import FLARE
import FLARE.filters
import matplotlib.pyplot as plt
import pandas as pd
from synphot import etau_madau
import matplotlib
import pickle
matplotlib.rcParams['font.size'] = (9)
matplotlib.rcParams['figure.figsize'] = (7.2,3.2)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
colors         = ['#e41a1c','#377eb8','#4daf4a','#984ea3',\
                  '#ff9f05','#ff7300','#ffbc05']#ffb405
                  #'#ff7f00','#ffe02e','#f781bf','#98ff98']*4

def load_quasar(filename):
    prop='trans_reflect'
    dt=pd.read_csv(filename,sep='\t',header=0,names=['lambda','inc_cont','trans_cont','emit_cont',\
  'net_cont','refl_cont','trans_reflect','refl_line','out_line','line_label','cont_label','num_lines'])
    dt=dt[(dt['lambda']>1e-3)&(dt['lambda']<1e1)]
    quasar_lambda=(dt['lambda'])
    quasar_nuLnu=(dt[prop])
    quasar_nu=3e8/(quasar_lambda*1e-6)
 
    ##Need to mock Lyman-forest extinction on the redshifted galaxy spectra
    wave=np.array(dt['lambda']*1e4*(1+z)) #angstrom
    extcurve = etau_madau(wave, z)
    extinct_q=extcurve(wave)
    extinct_q[wave<10**3.5]=0 #This has an upturn at low-lambda - manually get rid of this

    return np.log10(quasar_nuLnu/quasar_nu*extinct_q),quasar_lambda#*(1.+z))


def load_host(data):
    # --- now generate the various SEDs (nebular, stellar, total) [THIS IS IN THE REST-FRAME]
    o = models.generate_SED(model, data.Masses, data.Ages, data.Metallicities, (10**model.dust['A'])*data.MetSurfaceDensities, fesc = fesc)

    # --- now calculate some broad band photometry [THIS IS IN THE REST-FRAME]
    o.total.get_Lnu(F) # generates Lnu (broad band luminosities)
    o.total.get_fnu(cosmo, z) # generates lamz and fnu

    extcurve = etau_madau(o.lam*(1+z), z-0.01) #Slight shift as otherwise you cut off the Lyalpha line
    extinct_g=extcurve(o.lam*(1+z))
    extinct_g[o.lam*(1+z)<10**3]=0 #This has an upturn at low-lambda - manually get rid of this

    return np.log10(o.intrinsic_total.lnu*extinct_g), np.log10(o.total.lnu*extinct_g), o.lam/1e4


#def plot_images(BH,ax):
#  quasar_spectra=np.zeros(2766)

#  quasar_spectra, q_lambda = load_quasar(folder+str(BH)+'/run_cloudy.con')#,axes[jj,ii%cols])
  
#  ax.plot(q_lambda,quasar_spectra,label='Quasar (Intrinsic)',c='k',lw=1.5,zorder=100,alpha=0.2)
  #ax.plot(q_lambda,quasar_spectra+np.log10(dust_atten),label='Quasar',c='k',lw=1.5,zorder=100)

#  ax.set_xlabel(r'$\log_{10}(\lambda_{\rm{obs}}/\mu m)$')
#  return

def plot_images(BH,dust_atten,ax):
  quasar_spectra=np.zeros(2766)
  gal_spectra_tot=np.zeros(19999)
  gal_spectra_int=np.zeros(19999)

  data = SynthObs.bluetides_data('PIG_208/processed_data/'+str(BH))
  quasar_spectra, q_lambda = load_quasar(folder+str(BH)+'/run_cloudy.con')#,axes[jj,ii%cols])
  gal_spectra_int, gal_spectra_tot, g_lambda = load_host(data)
  
  ax.plot(q_lambda,quasar_spectra+np.log10(dust_atten),label='Quasar',c='k',lw=1.5,zorder=100)
  #ax[0].fill_between(q_lambda,quasar_spectra,quasar_spectra+np.log10(dust_atten[0]),color='k',alpha=0.15)

  ax.plot(g_lambda,gal_spectra_tot,label='Host Galaxy',c='turquoise',lw=1.5,zorder=80)
  #ax[0].fill_between(g_lambda,gal_spectra_tot,gal_spectra_int,color='turquoise',alpha=0.15)
  ax.set_xlabel(r'$\log_{10}(\lambda_{\rm{obs}}/\mu m)$')
  return



# ------ Plot spectra
folder='/home/mmarshal/BLUETIDES/BlueTides/PIG_208/processed_data/'


###NOTE: Need to extract these numbers from BH_spectra_z7_dust
BHsamples='SDSS_AGN_dust/9' 
tau_UV=1.165 #Min tau UV for MMBH, CO, WFIRST
dust_atten=np.exp(-np.array(tau_UV))#Need metallicity factor


#Setup
model = models.define_model('BPASSv2.2.1.binary/ModSalpeter_300') # DEFINE SED GRID - 
model.dust = {'A': 4.6, 'slope': -1.0} # DEFINE DUST MODEL - these are the calibrated z=8 values for the dust model
fesc = 0.9
filters = ['FAKE.FAKE.'+f for f in ['1500','2500','V']] # --- define the filters. FAKE.FAKE are just top-hat filters using for extracting rest-frame quantities.
cosmo = FLARE.default_cosmo()
z = 7.3

F = FLARE.filters.add_filters(filters, new_lam = model.lam) 


with open('wavelength.pkl', 'rb') as f:
        wavelength=pickle.load(f)

with open('detect_rate.pkl', 'rb') as f:
        detect_rate=pickle.load(f)

#Plot success rates
#fig,ax = plt.subplots(2,1,figsize=(4,4),gridspec_kw={'bottom':0.15,'left':0.15,'right':0.95,'top':0.95,'height_ratios':[2,1]},sharex=True)
fig,ax = plt.subplots(figsize=(3.6,3.2),gridspec_kw={'bottom':0.15,'left':0.15,'right':0.95,'top':0.85})
  
ax.plot(wavelength['1 ks'],detect_rate['1 ks'],'o',color=colors[6],label='1 ks')#,markerfacecolor='w',markeredgewidth=2.1)
ax.plot(2.0,detect_rate['2.5 ks'],'o',color=colors[4],label='2.5 ks')#,markerfacecolor='w',markeredgewidth=2.1)
ax.plot(wavelength['5 ks'],detect_rate['5 ks'],'o',color=colors[5],label='5 ks')#,markerfacecolor='None',markeredgewidth=2.1)

for filt in ['F115W','F150W','F277W','F356W','F444W']:
    ax.plot(wavelength[filt],detect_rate[filt],'o',color=colors[0],label='__nolabel__')#filt)
    ax.plot(wavelength[filt],detect_rate[filt+' 5 ks'],'o',color=colors[5],label='__nolabel__')#,markerfacecolor='None',markeredgewidth=2.1)#filt)
for filt in ['F356W']:
    ax.plot(wavelength[filt],detect_rate[filt],'o',color=colors[0],label='__nolabel__')#filt)
    ax.plot(wavelength[filt],detect_rate[filt+' 5 ks'],'o',color=colors[5],label='__nolabel__',markerfacecolor='None',markeredgewidth=2.1)#filt)
ax.plot(wavelength['F200W'],detect_rate['F200W'],'o',color=colors[0],label='10 ks')
  
ax.plot(wavelength['F150W 4800s'],detect_rate['F150W 4800s'],'o',color=colors[1],label='4.8 ks')
ax.plot(wavelength['HST'],detect_rate['HST'],'s',color=colors[1],label='4.8 ks (HST)')
ax.plot(wavelength['F560W'],detect_rate['F560W'],'^',color=colors[0],label='10 ks (MIRI)')

  
plt.legend(fontsize='small',loc=(0.26,0.02),ncol=2)
ax.set_xlabel(r'Observed Wavelength ($\mu$m)')
ax.set_ylabel('Fraction of Successful Detections')
ax.set_xlim([0.9,5.9])

ax2=ax.twiny()
wave=[2000,4000,6000] 
ax2.set_xticks(np.array(wave)/10000*8)
ax2.set_xticklabels(wave)
ax2.set_xlim(ax.get_xlim())
ax2.set_xlabel(r'Rest-Frame Wavelength ($\AA$)')
plt.savefig('success_rates_proposal.pdf')
  #plt.tight_layout()

"""cols=1


plot_images(BHsamples,dust_atten,ax[1])


ax[1].set_ylabel(r'$\log_{10}(L_{\nu}/{\rm{erg\ s}}^{-1}\ \rm{Hz}^{-1})$')
ax[1].set_ylim([28,32.2])
ax[1].legend(fontsize='small',loc=(0.3,-0.27),ncol=4)"""
plt.show()
