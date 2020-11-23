import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import matplotlib
matplotlib.rcParams['font.size'] = (9)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def flux_to_mag(flux, zp=0.0):
    return -2.5*np.log10(flux) + zp



df=pd.read_pickle('/home/mmarshal/BLUETIDES/BlueTides/PIG_208/processed_data/quasarDatabase_processed.pkl')
#df=pd.read_pickle('/home/mmarshal/BLUETIDES/BlueTides/PIG_208/processed_data/quasarDatabase.pkl')
df=df[df['Sample']=='SDSS']
print(df.columns)


fig1,ax1=plt.subplots(figsize=(3,3.1))
fig2,ax2=plt.subplots(2,2,figsize=(6,5.8),gridspec_kw={'hspace':0.3,'wspace':0.3})
#fig3,ax3=plt.subplots(1,2,figsize=(6,3.1),gridspec_kw={'hspace':0.3})
fig4,ax4=plt.subplots(figsize=(3,3.1))

for filt in ['F115W','F150W','F200W','F277W','F356W','F444W']:
    fit_mag=np.zeros(len(df))
    err_mag=np.zeros(len(df))
    fit_rad=np.zeros(len(df))
    err_rad=np.zeros(len(df))
    fit_sers=np.zeros(len(df))
    err_sers=np.zeros(len(df))
    if filt=='F200W':
      OH_mag=np.zeros(len(df))
      OH_mag_err=np.zeros(len(df))
      OH_rad=np.zeros(len(df))
      OH_sers=np.zeros(len(df))
      OH_rad_err=np.zeros(len(df))
      OH_sers_err=np.zeros(len(df))

    for jj,ii in enumerate(df['Index']):
        if ii!=41:
            if filt=='F200W':
                _pattern = 'runJWST/SDSS_z7_SN/mcmc_out_mock_JWST_SDSS_{}_residual.fits'
                _pattern_OH= 'runJWST/SDSS_z7_SN_onlyHost/mcmc_out_mock_JWST_SDSS_{}_residual.fits'
                head = fits.getheader(_pattern.format(ii))
                head_OH = fits.getheader(_pattern_OH.format(ii))
                OH_mag[jj]=head_OH['1SER_MAG'].split(' ')[0]
                OH_mag_err[jj]=head_OH['1SER_MAG'].split(' ')[-1]
                OH_rad[jj]=head_OH['1SER_RE'].split(' ')[0]
                OH_sers[jj]=head_OH['1SER_N'].split(' ')[0]
                OH_rad_err[jj]=head_OH['1SER_RE'].split(' ')[-1]
                OH_sers_err[jj]=head_OH['1SER_N'].split(' ')[-1]
            else:
                _pattern = 'runJWST/SDSS_z7_{}/mcmc_out_mock_JWST_SDSS_{}_residual.fits'
                head = fits.getheader(_pattern.format(filt,ii))
            fit_mag[jj]=head['2SER_MAG'].split(' ')[0]
            err_mag[jj]=head['2SER_MAG'].split(' ')[-1]
            fit_rad[jj]=head['2SER_RE'].split(' ')[0]
            err_rad[jj]=head['2SER_RE'].split(' ')[-1]
            fit_sers[jj]=head['2SER_N'].split(' ')[0]
            err_sers[jj]=head['2SER_N'].split(' ')[-1]
    
    if filt in ['F277W','F356W','F444W']:
      pxscale = 0.063/2 #arcsec
    else:
      pxscale = 0.031/2
    flux = df['{} Flux'.format(filt)]
    mag = flux_to_mag(flux,25.9463)
    
    df_T = df[df[filt]==True]
    #df_F = df[df[filt]==False]
    #ax1.plot(df_T['{} True Mag'.format(filt)],df_T['{} Fit Mag'.format(filt)],'o',label=filt)
    ax1.errorbar(mag[df[filt]==True],fit_mag[df[filt]==True],err_mag[df[filt]==True],fmt='o',label=filt)
    #ax2[0].plot(df_T['Radius'],fit_rad[df[filt]==True]*pxscale,'o',label=filt)
    ax2[0,0].errorbar(df_T['Radius'],fit_rad[df[filt]==True]*pxscale,err_rad[df[filt]==True]*pxscale,fmt='o',label=filt)
    #ax2.plot(df_F['Radius'],fit_rad[df[filt]==False]*pxscale,'o',markerfacecolor='w')
    #ax2[1].plot(df_T['BtoT'],fit_sers[df[filt]==True],'o',label=filt)
    ax2[1,0].errorbar(df_T['BtoT'],fit_sers[df[filt]==True],err_sers[df[filt]==True],fmt='o',label=filt)
    #plt.plot(df_F['{} True Mag'.format(filt)],df_F['{} Fit Mag'.format(filt)],'o',markerfacecolor='w')
    print(filt)
    print(np.mean(fit_mag[df[filt]==True]-mag[df[filt]==True])) 
    print(np.max(fit_mag[df[filt]==True]-mag[df[filt]==True]))
    print('Radius',np.corrcoef(df_T['Radius'],fit_rad[df[filt]==True]*pxscale)[0,1])
    print('BtoT',np.corrcoef(df_T['BtoT'],fit_sers[df[filt]==True])[0,1])

    ###Compare only host fit to subtracted fit
    if filt=='F200W':
      #ax3[0].plot(OH_mag[df[filt]==True],fit_mag[df[filt]==True],'o')
      #ax3[0].plot(OH_rad[df[filt]==True]*pxscale,fit_rad[df[filt]==True]*pxscale,'o',color='C2')
      #ax3[1].plot(OH_sers[df[filt]==True],fit_sers[df[filt]==True],'o',color='C2')
      ax2[0,1].errorbar(OH_rad[df[filt]==True]*pxscale,fit_rad[df[filt]==True]*pxscale,yerr=err_rad[df[filt]==True]*pxscale,xerr=OH_rad_err[df[filt]==True]*pxscale,fmt='o',color='C2')
      ###Plot undetectable quasars separately
      #locc=df.index.isin([0,4,10,12,21,22])
      #ax2[0,1].errorbar(OH_rad[locc]*pxscale,fit_rad[locc]*pxscale,yerr=err_rad[locc]*pxscale,xerr=OH_rad_err[locc]*pxscale,fmt='o',color='C3')
      ax2[1,1].errorbar(OH_sers[df[filt]==True],fit_sers[df[filt]==True],yerr=err_sers[df[filt]==True],xerr=OH_sers_err[df[filt]==True],fmt='o',color='C2')
      ax4.errorbar(OH_mag[df[filt]==True],fit_mag[df[filt]==True],yerr=err_mag[df[filt]==True],xerr=OH_mag_err[df[filt]==True],fmt='o',color='C2')
      print('mean',np.mean(fit_mag[df[filt]==True]-OH_mag[df[filt]==True])) 
      print('max (+ve)',np.max(fit_mag[df[filt]==True]-OH_mag[df[filt]==True]))
      print('max (-ve)',np.min(fit_mag[df[filt]==True]-OH_mag[df[filt]==True]))

ax1.plot([22.5,27],[22.5,27],'k')
ax1.invert_xaxis()
ax1.invert_yaxis()
ax1.set_xlim([26.4,22.8])
ax1.set_ylim([26.4,22.8])
ax1.legend(fontsize='small')
ax1.set_ylabel('Sersic Magnitude, PSF Subtraction')
ax1.set_xlabel('True Magnitude')


ax2[0,0].legend(fontsize='small')
ax2[0,0].set_ylabel('Sersic Radius (kpc), PSF Subtraction')
ax2[0,0].set_xlabel('Half-Mass Radius (kpc)')

ax2[1,0].set_ylabel('Sersic Index, PSF Subtraction')
ax2[1,0].set_xlabel('Bulge-to-Total Mass Ratio')

ax2[0,1].plot([0.10,0.17],[0.10,0.17],'k')
ax2[1,1].plot([0,5.5],[0,5.5],'k')

ax2[0,1].set_xlim([0.11,0.17])
ax2[0,1].set_ylim([0.11,0.17])
ax2[1,1].set_xlim([0,5.5])
ax2[1,1].set_ylim([0,5.5])
ax2[0,1].set_ylabel('Sersic Radius (kpc), PSF Subtraction')
ax2[0,1].set_xlabel('Sersic Radius (kpc), True Host Image')
ax2[1,1].set_ylabel('Sersic Index, PSF Subtraction')
ax2[1,1].set_xlabel('Sersic Index, True Host Image')

ax4.plot([22.5,27],[22.5,27],'k')
ax4.invert_xaxis()
ax4.invert_yaxis()
ax4.set_xlim([26.4,22.8])
ax4.set_ylim([26.4,22.8])
ax4.set_xlabel('Sersic Magnitude, True Host Image')
ax4.set_ylabel('Sersic Magnitude, PSF Subtraction')

fig1.subplots_adjust(left=0.2, bottom=0.15, right=0.95, top=0.95)
fig4.subplots_adjust(left=0.2, bottom=0.15, right=0.95, top=0.95)
fig2.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95)
#fig3.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.9)

fig1.savefig('compare_magnitudes.pdf')
fig2.savefig('compare_measured_props.pdf')
#fig3.savefig('compare_fit_props.pdf')
fig4.savefig('compare_fit_mags.pdf')


plt.show()

