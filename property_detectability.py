import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from photutils import CircularAperture,CircularAnnulus
from photutils import aperture_photometry, background
from matplotlib.colors import SymLogNorm
import sys
import pandas as pd
import calculate_BIC
import matplotlib
from scipy import stats
matplotlib.rcParams['font.size'] = (8)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def hist_x(data,mask,ax,**kwargs):
    nbins=4
    frac=np.zeros(nbins)
    err=np.zeros(nbins)
    err_up=np.zeros(nbins)
    bins_min=np.amin(data)-0.01
    bins_max=np.amax(data)
    bins_width=(bins_max-bins_min)/nbins
    for bb in range(0,nbins):
        b_min=bins_min+bb*bins_width
        b_max=bins_min+(bb+1)*bins_width
        if len(data[(data<=b_max) & (data>b_min)])>0:
          frac[bb]=len(data[mask][(data[mask]<=b_max) & (data[mask]>b_min)])/len(data[(data<=b_max) & (data>b_min)])
          #err[bb]=1/np.sqrt(len(data[(data<=b_max) & (data>b_min)]))
          ninbin=len(data[(data<=b_max) & (data>b_min)])
          zscore=1.96
          w_min = (2*ninbin*frac[bb]+zscore**2-(zscore*np.sqrt(zscore**2-1/ninbin+4*ninbin*frac[bb]*(1-frac[bb])+(4*frac[bb]-2))+1))/(2*(ninbin+zscore**2))
          w_min = np.max([0,w_min])
          w_max = (2*ninbin*frac[bb]+zscore**2+(zscore*np.sqrt(zscore**2-1/ninbin+4*ninbin*frac[bb]*(1-frac[bb])-(4*frac[bb]-2))+1))/(2*(ninbin+zscore**2))
          w_max = np.min([1,w_max])

          err[bb]=frac[bb]-w_min
          err_up[bb]=w_max-frac[bb]
          if frac[bb]==1:
             err_up[bb]=0
          elif frac[bb]==0:
             err[bb]=0
        else:
          frac[bb]=np.nan
          err[bb]=0
    ax.bar(bins_min+bins_width*(0.5+np.arange(0,nbins)),height=frac,width=bins_width,edgecolor='k',facecolor='#d6f1ff',yerr=[err,err_up],capsize=3,ecolor=[0.5,0.5,0.5])#[0.8,0.8,0.8])
    #ax.plot(bins_min+bins_width*(1/2+np.arange(0,6)),frac,'ro',zorder=99)#, linewidth=1.2,marker='o',**kwargs)
    ax.set_xlim(bins_min-0.05,bins_max+0.05)
    ax.set_ylim(-0.05,1.171)
    ax.set_yticks([0,1])
    ax.set_yticklabels(['0',''])
    ax.tick_params(axis='both', direction='in')
    return

def hist_y(data,mask,ax,**kwargs):
    nbins=4
    frac=np.zeros(nbins)
    err=np.zeros(nbins)
    err_up=np.zeros(nbins)
    bins_min=np.amin(data)-0.01
    bins_max=np.amax(data)
    bins_width=(bins_max-bins_min)/nbins
    for bb in range(0,nbins):
        b_min=bins_min+bb*bins_width
        b_max=bins_min+(bb+1)*bins_width
        if len(data[(data<=b_max) & (data>b_min)])>0:
          frac[bb]=len(data[mask][(data[mask]<=b_max) & (data[mask]>b_min)])/len(data[(data<=b_max) & (data>b_min)])
          #err[bb]=np.sqrt(len(data[(data<=b_max) & (data>b_min)]))/len(data[(data<=b_max) & (data>b_min)])
          #err[bb]=1/np.sqrt(len(data[(data<=b_max) & (data>b_min)]))
          ninbin=len(data[(data<=b_max) & (data>b_min)])
          #t_value=stats.t.ppf(1-0.1589,ninbin)
          #err[bb]=t_value*np.sqrt((frac[bb]*(1-frac[bb]))/ninbin)
          ###WILSON SCORE INTERVAL
          zscore=1.96
          w_min = (2*ninbin*frac[bb]+zscore**2-(zscore*np.sqrt(zscore**2-1/ninbin+4*ninbin*frac[bb]*(1-frac[bb])+(4*frac[bb]-2))+1))/(2*(ninbin+zscore**2))
          w_min = np.max([0,w_min])
          w_max = (2*ninbin*frac[bb]+zscore**2+(zscore*np.sqrt(zscore**2-1/ninbin+4*ninbin*frac[bb]*(1-frac[bb])-(4*frac[bb]-2))+1))/(2*(ninbin+zscore**2))
          w_max = np.min([1,w_max])

          err[bb]=frac[bb]-w_min
          err_up[bb]=w_max-frac[bb]
          if frac[bb]==1:
             err_up[bb]=0
          elif frac[bb]==0:
             err[bb]=0
          #if frac[bb]==1:
          #   err[bb]=1.14/ninbin
          #   err_up[bb]=0
          #elif frac[bb]==0:
          #   if ninbin>1:
          #     err_up[bb]=1.14/ninbin
          #   else: 
          #     err_up[bb]=1
          #   err[bb]=0
          #if frac[bb]+err[bb]>1:
          #   err_up[bb]=1-frac[bb]
          #else:
          #   err_up[bb]=err[bb]
        else:
          frac[bb]=np.nan
          err[bb]=0
    ax.barh(bins_min+bins_width*(0.5+np.arange(0,nbins)),width=frac,height=bins_width,edgecolor='k',facecolor='#d6f1ff',xerr=[err,err_up],capsize=3,ecolor=[0.5,0.5,0.5])#[0.8,0.8,0.8])
    #ax.plot(frac,bins_min+bins_width*(1/2+np.arange(0,6)),'ro',zorder=99)#, linewidth=1.2,marker='o',**kwargs)
    ax.set_ylim(bins_min-0.02,bins_max+0.02)
    ax.set_xlim(-0.05,1.135)
    ax.set_xticks([0,1])
    ax.tick_params(axis='both', direction='in')
    return

def three_pane_plot(xdata,ydata,mask,ax,xlim=0.05,ylim=0.01):
    ax[0,1].plot(xdata[easy],ydata[easy],'ko',label='Detectable')#,color='#984ea3')         #['#e41a1c','#377eb8','#4daf4a','#984ea3')
    ax[0,1].plot(xdata[~easy],ydata[~easy],'ko',markerfacecolor='w',label='Undetectable')#,color='#984ea3')
    hist_x(xdata,easy,ax[1,1])
    hist_y(ydata,easy,ax[0,0])
    
    bins_min=np.amin(xdata)
    bins_max=np.amax(xdata)
    ax[0,1].set_xlim(bins_min-xlim,bins_max+xlim)
    ax[1,1].set_xlim(bins_min-xlim,bins_max+xlim)
    bins_min=np.amin(ydata)
    bins_max=np.amax(ydata)
    ax[0,1].set_ylim(bins_min-ylim,bins_max+ylim)
    ax[0,0].set_ylim(bins_min-ylim,bins_max+ylim)
    ax[0,1].set_yticks([])
    ax[0,1].set_xticks([])
    ax[1,0].axis('off')
    return

if __name__=='__main__':
    title='SDSS'
    indices = [2,   3,   6,   7,   8,   9,  10,  12,  16,  18,  20,  22,  23,  25,  27,  32,  36,  40,   43,  45,  46, 100]
    
    df=pd.read_pickle('/home/mmarshal/BLUETIDES/BlueTides/PIG_208/processed_data/quasarDatabase_processed.pkl')
    df=df[df['Sample']==title]
    print(df.columns)
    BHAR=[14.500942,  13.576554,  10.321727,   8.939834,   8.855372,   8.808868,\
  8.62038,    8.601731,   7.5696898,  7.16324,    6.7052402,  6.6734014,\
  6.5346804,  6.1150928,  5.7489934,  5.2159667,  4.8983135,  4.826857,\
  4.396038,   4.2712193,  4.2201514,  2.0458283]
    EddRat=[2.08884382, 2.08884263, 2.08884468, 1.49570326, 1.73184892, 1.97549235,\
 2.08883628, 2.08884432, 2.08884368, 2.08883966, 1.45465988, 1.41140228,\
 1.28216163, 2.08883995, 0.72223113, 2.0304664,  2.08883499, 0.7652108,\
 0.75407579, 1.34089491, 0.93656116, 0.66053557]


    detectability=np.zeros(len(indices)) #detectable in how many filters?
    
    for filt in ['F115W','F150W','F200W','F277W','F356W','F444W']:
      detectability+=np.array(df[filt],dtype=int)
    easy=detectability>4
    #print(df[easy])
    #print(df[~easy])

    if True: #Full-pane galaxy properties
    
      fig,ax=plt.subplots(3,2,figsize=(3.2,4.8),gridspec_kw={'width_ratios':[0.3,1],'height_ratios':[1,1,0.3],'hspace':0,'wspace':0,'bottom':0.07,'left':0.2,'right':0.95,'top':0.98})   
    
      labs={'BHMass':r'$\log(M_{\rm{BH}}/M_\odot)$','SFR':r'$\log(\rm{SFR}/M_\odot \rm{yr}^{-1})$'}
      for ii,prop in enumerate(['SFR']): 
        #fig,ax=plt.subplots(1,2,figsize=(4,3),gridspec_kw={'width_ratios':[0.3,1],'wspace':0,'bottom':0.2})   
 
        ax[ii,1].plot(np.log10(df['StellarMass'])[easy],np.log10(df[prop])[easy],'ko')
        ax[ii,1].plot(np.log10(df['StellarMass'])[~easy],np.log10(df[prop])[~easy],'ko',markerfacecolor='w')
        hist_y(np.log10(df[prop]),easy,ax[ii,0])
    
        bins_min=np.amin(np.log10(df['StellarMass']))
        bins_max=np.amax(np.log10(df['StellarMass']))
        ax[ii,1].set_xlim(bins_min-0.05,bins_max+0.04)
        bins_min=np.amin(np.log10(df[prop]))
        bins_max=np.amax(np.log10(df[prop]))
        #ax[ii,1].set_ylim(bins_min-0.05,bins_max+0.05)
        ax[ii,1].set_ylim(bins_min-0.02,bins_max+0.02)
        ax[ii,1].set_yticks([])
        ax[ii,0].set_ylabel(labs[prop])
        #ax[ii,1].set_xlabel(r'$\log(M_\ast/M_\odot)$')
    
      ii+=1
      for prop in ['BtoT']:#'Radius']: 
        #fig,ax=plt.subplots(2,2,figsize=(4,4),gridspec_kw={'width_ratios':[0.3,1],'height_ratios':[1,0.3],'hspace':0,'wspace':0,'bottom':0.2,'left':0.2,'right':0.95})   
 
        ax[ii,1].plot(np.log10(df['StellarMass'])[easy],(df[prop])[easy],'ko',label='Detectable')
        ax[ii,1].plot(np.log10(df['StellarMass'])[~easy],(df[prop])[~easy],'ko',markerfacecolor='w',label='Undetectable')
        hist_x(np.log10(df['StellarMass']),easy,ax[ii+1,1],color='k')
        ax[ii+1,1].set_ylim(-0.05,1.14)
        ax[ii+1,1].set_yticklabels(['0',''])
        hist_y(df[prop],easy,ax[ii,0],color='k')
    
        bins_min=np.amin(np.log10(df['StellarMass']))
        bins_max=np.amax(np.log10(df['StellarMass']))
        ax[ii,1].set_xlim(bins_min-0.05,bins_max+0.04)
        ax[ii+1,1].set_xlim(bins_min-0.05,bins_max+0.04)
        bins_min=np.amin((df[prop]))
        bins_max=np.amax((df[prop]))
        #ax[2,1].set_ylim(bins_min-0.05,bins_max+0.05)
        ax[ii,1].set_ylim(bins_min-0.02,bins_max+0.02)
        ax[ii,1].set_yticks([])
        ax[ii,1].set_xticks([])
        ax[ii,0].set_ylabel(r'$M_{\rm{bulge}}/M_\ast$')
        ax[ii+1,1].set_xlabel(r'$\log(M_\ast/M_\odot)$')
        ax[ii,0].set_xlabel('Success\nRate')
        ax[ii,1].legend()
        ax[ii+1,0].axis('off')
        plt.savefig('properties_stellarMass.pdf') 
   

    if True: #Radius 3-pane
      OH_rad=np.zeros(len(df))
      for jj,ii in enumerate(df['Index']):
        if ii!=41:
          #_pattern_OH= 'runJWST/SDSS_z7_SN_onlyHost/mcmc_out_mock_JWST_SDSS_{}_residual.fits'
          _pattern_OH= 'runJWST/SDSS_z7_F277W/mcmc_out_mock_JWST_SDSS_{}_residual.fits'
          head_OH = fits.getheader(_pattern_OH.format(ii))
          OH_rad[jj]=head_OH['2SER_RE'].split(' ')[0]
      
      if filt in ['F277W','F356W','F444W']:
        pxscale = 0.063/2 #arcsec
      else:
        pxscale = 0.031/2
     
      pxscale *= 0.269/0.05 #to kpc
    
      OH_rad*=pxscale

      xdata=df['Radius']
      ydata=OH_rad
    
      fig,ax=plt.subplots(2,2,figsize=(3.2,2.8),gridspec_kw={'width_ratios':[0.3,1],'height_ratios':[1,0.3],'hspace':0,'wspace':0,'bottom':0.15,'left':0.2,'right':0.95,'top':0.95})   
      three_pane_plot(xdata,ydata,easy,ax,0.025,0.025)
      ax[1,1].set_xlabel(r'$R_{0.5}/\rm{kpc}$')
      ax[0,0].set_ylabel(r'F277W Sersic Radius (kpc)')
      ax[0,0].set_xlabel('Success\nRate')
      #ax[0,1].legend(loc='lower right')
      plt.savefig('properties_radius.pdf') 
      plt.show()

      #fig,ax=plt.subplots(2,3)
    
      #ax[1,0].plot(df['MUV_AGN'][easy]+df['tau_UV_AGN'][easy],df['MUV_gal_dust'][easy],'ko')
      #ax[1,0].plot(df['MUV_AGN'][~easy]+df['tau_UV_AGN'][~easy],df['MUV_gal_dust'][~easy],'ro')
      #ax[1,2].plot(df['MUV_AGN']+df['tau_UV_AGN']-df['MUV_gal_dust'],detectability,'ko')
      #plt.show()
    
    if True: #Magnitude 3-pane
      xdata=df['MUV_AGN_dust']
      ydata=df['MUV_gal_dust']
    
      fig,ax=plt.subplots(2,2,figsize=(3.2,2.8),gridspec_kw={'width_ratios':[0.3,1],'height_ratios':[1,0.3],'hspace':0,'wspace':0,'bottom':0.15,'left':0.2,'right':0.95,'top':0.95})   
      three_pane_plot(xdata,ydata,easy,ax,0.2,0.2)
      ax[1,1].set_xlabel(r'$M_{\rm{UV,~ AGN}}$')
      ax[0,0].set_ylabel(r'$M_{\rm{UV,~ host}}$')
      ax[0,0].set_xlabel('Success\nRate')
      ax[0,0].invert_yaxis()
      ax[1,1].invert_xaxis()
      ax[0,1].invert_yaxis()
      ax[0,1].invert_xaxis()
      plt.savefig('properties_magnitudes_dust.pdf') 
      plt.show()
  
 
    if True: #MAGN vs Mgal and Mgal-MAGN
      fig,ax=plt.subplots(3,2,figsize=(3.2,4.8),gridspec_kw={'width_ratios':[0.3,1],'height_ratios':[1,1,0.3],'hspace':0,'wspace':0,'bottom':0.15,'left':0.2,'right':0.95,'top':0.98})   
     
      xdata=df['MUV_AGN_dust']
      ydata=df['MUV_gal_dust']

      ax[0,1].plot(xdata[easy],ydata[easy],'ko',label='Detectable')
      ax[0,1].plot(xdata[~easy],ydata[~easy],'ko',markerfacecolor='w',label='Undetectable')
      hist_x(xdata,easy,ax[1,1],color='k')
      #ax[2,1].set_ylim(-0.05,1.14)
      hist_y(ydata,easy,ax[0,0],color='k')
    
      bins_min=np.amin(xdata)
      bins_max=np.amax(xdata)
      ax[0,1].set_xlim(bins_min-0.1,bins_max+0.1)
      ax[1,1].set_xlim(bins_min-0.1,bins_max+0.1)
      bins_min=np.amin(ydata)
      bins_max=np.amax(ydata)
      #ax[2,1].set_ylim(bins_min-0.05,bins_max+0.05)
      ax[0,1].set_ylim(bins_min-0.1,bins_max+0.1)
      ax[0,1].set_yticks([])
      ax[0,0].set_yticks([-22.5,-22,-21.5,-21,-20.5])
      ax[0,1].set_xticks([])
      ax[0,0].set_ylabel(r'$M_{\rm{UV,~ host}}$')
 
      ydata=df['MUV_gal_dust']-df['MUV_AGN_dust']

      ax[1,1].plot(xdata[easy],ydata[easy],'ko',label='Detectable')
      ax[1,1].plot(xdata[~easy],ydata[~easy],'ko',markerfacecolor='w',label='Undetectable')
      hist_x(xdata,easy,ax[2,1],color='k')
      ax[2,1].set_ylim(-0.05,1.14)
      ax[2,1].set_yticklabels(['0',''])
      hist_y(ydata,easy,ax[1,0],color='k')
   
      bins_min=np.amin(xdata)
      bins_max=np.amax(xdata)
      ax[1,1].set_xlim(bins_min-0.1,bins_max+0.1)
      ax[2,1].set_xlim(bins_min-0.1,bins_max+0.1)
      bins_min=np.amin(ydata)
      bins_max=np.amax(ydata)
      #ax[2,1].set_ylim(bins_min-0.05,bins_max+0.05)
      ax[1,1].set_ylim(bins_min-0.1,bins_max+0.1)
      ax[1,1].set_yticks([])
      ax[1,1].set_xticks([])
      ax[2,1].set_xlabel(r'$M_{\rm{UV,~ AGN}}$')
      ax[1,0].set_ylabel(r'$M_{\rm{UV,~ host}}-M_{\rm{UV,~ AGN}}$')
      ax[1,0].set_xlabel('Success\nRate')
      ax[1,1].legend()
      ax[0,0].invert_yaxis()
      ax[0,1].invert_yaxis()
      ax[0,1].invert_xaxis()
      ax[1,1].invert_xaxis()
      ax[2,1].invert_xaxis()
      ax[2,0].axis('off')
      plt.savefig('properties_magnitudes_dust_difference.pdf') 
      plt.show()
   
    if True:#BHAR,Eddington ratio
    
      fig,ax=plt.subplots(3,2,figsize=(3.2,4.8),gridspec_kw={'width_ratios':[0.3,1],'height_ratios':[1,1,0.3],'hspace':0,'wspace':0,'bottom':0.1,'left':0.2,'right':0.95,'top':0.98})   
    
      xdata=np.log10(df['BHMass'])
      ydata=np.array(BHAR)
 
      ax[0,1].plot(xdata[easy],ydata[easy],'ko',label='Detectable')
      ax[0,1].plot(xdata[~easy],ydata[~easy],'ko',markerfacecolor='w',label='Undetectable')
      #ax[2,1].set_ylim(-0.05,1.14)
      hist_y(ydata,easy,ax[0,0],color='k')
    
      bins_min=np.amin(xdata)
      bins_max=np.amax(xdata)
      ax[0,1].set_xlim(bins_min-0.1,bins_max+0.1)
      ax[1,1].set_xlim(bins_min-0.1,bins_max+0.1)
      bins_min=np.amin(ydata)
      bins_max=np.amax(ydata)
      #ax[2,1].set_ylim(bins_min-0.05,bins_max+0.05)
      ax[0,1].set_ylim(bins_min-0.5,bins_max+0.5)
      ax[0,1].set_yticks([])
      ax[0,1].set_xticks([])
      ax[0,0].set_ylabel(r'$\rm{BHAR}~ (M_\odot/\rm{yr})$')
 
      ydata=np.array(EddRat)

      ax[1,1].plot(xdata[easy],ydata[easy],'ko',label='Detectable')
      ax[1,1].plot(xdata[~easy],ydata[~easy],'ko',markerfacecolor='w',label='Undetectable')
      hist_x(xdata,easy,ax[2,1],color='k')
      ax[2,1].set_ylim(-0.05,1.152)
      ax[2,1].set_yticklabels(['0',''])
      hist_y(ydata,easy,ax[1,0],color='k')
   
      bins_min=np.amin(xdata)
      bins_max=np.amax(xdata)
      ax[1,1].set_xlim(bins_min-0.1,bins_max+0.1)
      ax[2,1].set_xlim(bins_min-0.1,bins_max+0.1)
      bins_min=np.amin(ydata)
      bins_max=np.amax(ydata)
      #ax[2,1].set_ylim(bins_min-0.05,bins_max+0.05)
      ax[1,1].set_ylim(bins_min-0.1,bins_max+0.1)
      ax[1,1].set_yticks([])
      ax[1,1].set_xticks([])
      ax[2,1].set_xlabel(r'$\log(M_{\rm{BH}}/M_\odot)$')
      ax[1,0].set_ylabel(r'$L_{\rm{Bol}}/L_{\rm{Edd}}$')
      ax[1,0].set_xlabel('Success\nRate')
      ax[0,1].legend()
      ax[2,0].axis('off')
      plt.savefig('properties_BHmassAccretion.pdf') 
      plt.show()
 
    if True:#intrinsic MAGN-Mgal
      xdata=df['MUV_AGN_dust']-1.086*df['tau_UV_AGN']
      ydata=df['MUV_gal']
    
      fig,ax=plt.subplots(2,2,figsize=(3.2,2.8),gridspec_kw={'width_ratios':[0.3,1],'height_ratios':[1,0.3],'hspace':0,'wspace':0,'bottom':0.15,'left':0.2,'right':0.95,'top':0.95})   
 
      three_pane_plot(xdata,ydata,easy,ax,0.08,0.08)
      ax[1,1].set_xlabel(r'$M_{\rm{UV,~ AGN~ (intrinsic)}}$')
      ax[0,0].set_ylabel(r'$M_{\rm{UV,~ host~ (intrinsic)}}$') 
      ax[0,0].invert_yaxis()
      ax[1,1].invert_xaxis()
      ax[0,1].invert_yaxis()
      ax[0,1].invert_xaxis()
      ax[0,0].set_xlabel('Success\nRate')
      #ax[0,1].legend(loc='lower right')
      plt.savefig('properties_magnitudes_intrinsic.pdf') 
   
    plt.show()
