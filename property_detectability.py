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
matplotlib.rcParams['font.size'] = (9)
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
          err[bb]=1/np.sqrt(len(data[(data<=b_max) & (data>b_min)]))
          if frac[bb]+err[bb]>1:
             err_up[bb]=1-frac[bb]
          else:
             err_up[bb]=err[bb]
        else:
          frac[bb]=np.nan
          err[bb]=0
    ax.bar(bins_min+bins_width*(0.5+np.arange(0,nbins)),height=frac,width=bins_width,edgecolor='k',facecolor='#d6f1ff',yerr=[err,err_up],capsize=3,ecolor=[0.5,0.5,0.5])#[0.8,0.8,0.8])
    #ax.plot(bins_min+bins_width*(1/2+np.arange(0,6)),frac,'ro',zorder=99)#, linewidth=1.2,marker='o',**kwargs)
    ax.set_xlim(bins_min-0.05,bins_max+0.05)
    ax.set_ylim(-0.05,1.171)
    ax.set_yticks([0,1])
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
          err[bb]=1/np.sqrt(len(data[(data<=b_max) & (data>b_min)]))
          if frac[bb]+err[bb]>1:
             err_up[bb]=1-frac[bb]
          else:
             err_up[bb]=err[bb]
        else:
          frac[bb]=np.nan
          err[bb]=0
    ax.barh(bins_min+bins_width*(0.5+np.arange(0,nbins)),width=frac,height=bins_width,edgecolor='k',facecolor='#d6f1ff',xerr=[err,err_up],capsize=3,ecolor=[0.5,0.5,0.5])#[0.8,0.8,0.8])
    #ax.plot(frac,bins_min+bins_width*(1/2+np.arange(0,6)),'ro',zorder=99)#, linewidth=1.2,marker='o',**kwargs)
    #ax.set_ylim(bins_min-0.05,bins_max+0.05)
    ax.set_xlim(-0.05,1.14)
    ax.set_xticks([0,1])
    ax.tick_params(axis='both', direction='in')
    return

def three_pane_plot(xdata,ydata,mask,ax):
    ax[0,1].plot(xdata[easy],ydata[easy],'ko',label='Detectable')#,color='#984ea3')         #['#e41a1c','#377eb8','#4daf4a','#984ea3')
    ax[0,1].plot(xdata[~easy],ydata[~easy],'ko',markerfacecolor='w',label='Undetectable')#,color='#984ea3')
    hist_x(xdata,easy,ax[1,1])
    hist_y(ydata,easy,ax[0,0])
    
    bins_min=np.amin(xdata)
    bins_max=np.amax(xdata)
    ax[0,1].set_xlim(bins_min-0.05,bins_max+0.05)
    bins_min=np.amin(ydata)
    bins_max=np.amax(ydata)
    ax[0,1].set_ylim(bins_min-0.01,bins_max+0.01)
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

    detectability=np.zeros(len(indices)) #detectable in how many filters?
    
    for filt in ['F115W','F150W','F200W','F277W','F356W','F444W']:
      detectability+=np.array(df[filt],dtype=int)

    """fig,ax=plt.subplots(2,4)   
 
    prop=['StellarMass','BHMass']#'SFR    Radius
    for ii,pp in enumerate(prop):
      ax[0,ii].plot(np.log10(df[pp]),detectability,'ko')

    #prop=['SFR', 'Radius', 'MUV_gal','MUV_gal_dust', 'MUV_AGN']
    prop=['SFR', 'Radius','MUV_gal_dust']
    for ii,pp in enumerate(prop):
      if ii<2:
        ax[0,ii+2].plot(df[pp],detectability,'ko')
        ax[0,ii+2].set_xlabel(pp)
      else:
        ax[1,ii-2].plot(df[pp],detectability,'ko')
        ax[1,ii-2].set_xlabel(pp)

    ax[1,1].plot(df['MUV_AGN']+df['tau_UV_AGN'],detectability,'ko')
    ax[1,2].plot(df['MUV_AGN']+df['tau_UV_AGN']-df['MUV_gal_dust'],detectability,'ko')
    plt.show()"""

    easy=detectability>4
    print(df[easy])
    print(df[~easy])
    
    fig,ax=plt.subplots(4,2,figsize=(4,9.5),gridspec_kw={'width_ratios':[0.3,1],'height_ratios':[1,1,1,0.3],'hspace':0,'wspace':0,'bottom':0.05,'left':0.2,'right':0.95,'top':0.98})   
    
    labs={'BHMass':r'$\log(M_{\rm{BH}}/M_\odot)$','SFR':r'$\log(\rm{SFR}/M_\odot \rm{yr}^{-1})$'}
    for ii,prop in enumerate(['BHMass','SFR']): 
      #fig,ax=plt.subplots(1,2,figsize=(4,3),gridspec_kw={'width_ratios':[0.3,1],'wspace':0,'bottom':0.2})   
 
      ax[ii,1].plot(np.log10(df['StellarMass'])[easy],np.log10(df[prop])[easy],'ko')
      ax[ii,1].plot(np.log10(df['StellarMass'])[~easy],np.log10(df[prop])[~easy],'ko',markerfacecolor='w')
      hist_y(np.log10(df[prop]),easy,ax[ii,0])
    
      bins_min=np.amin(np.log10(df['StellarMass']))
      bins_max=np.amax(np.log10(df['StellarMass']))
      ax[ii,1].set_xlim(bins_min-0.05,bins_max+0.05)
      bins_min=np.amin(np.log10(df[prop]))
      bins_max=np.amax(np.log10(df[prop]))
      #ax[ii,1].set_ylim(bins_min-0.05,bins_max+0.05)
      ax[ii,1].set_ylim(bins_min-0.01,bins_max+0.01)
      ax[ii,1].set_yticks([])
      ax[ii,0].set_ylabel(labs[prop])
      #ax[ii,1].set_xlabel(r'$\log(M_\ast/M_\odot)$')
    
    for prop in ['BtoT']:#'Radius']: 
      #fig,ax=plt.subplots(2,2,figsize=(4,4),gridspec_kw={'width_ratios':[0.3,1],'height_ratios':[1,0.3],'hspace':0,'wspace':0,'bottom':0.2,'left':0.2,'right':0.95})   
 
      ax[2,1].plot(np.log10(df['StellarMass'])[easy],(df[prop])[easy],'ko',label='Detectable')
      ax[2,1].plot(np.log10(df['StellarMass'])[~easy],(df[prop])[~easy],'ko',markerfacecolor='w',label='Undetectable')
      hist_x(np.log10(df['StellarMass']),easy,ax[3,1],color='k')
      hist_y(df[prop],easy,ax[2,0],color='k')
    
      bins_min=np.amin(np.log10(df['StellarMass']))
      bins_max=np.amax(np.log10(df['StellarMass']))
      ax[2,1].set_xlim(bins_min-0.05,bins_max+0.05)
      bins_min=np.amin((df[prop]))
      bins_max=np.amax((df[prop]))
      #ax[2,1].set_ylim(bins_min-0.05,bins_max+0.05)
      ax[2,1].set_ylim(bins_min-0.01,bins_max+0.01)
      ax[2,1].set_yticks([])
      ax[2,1].set_xticks([])
      ax[2,0].set_ylabel(r'$M_{\rm{bulge}}/M_\ast$')
      ax[3,1].set_xlabel(r'$\log(M_\ast/M_\odot)$')
      ax[2,0].set_xlabel('Success\nRate')
      ax[2,1].legend()
    plt.savefig('properties_stellarMass.pdf') 
   

    """
    fig,ax=plt.subplots(3,2,figsize=(4,5),gridspec_kw={'width_ratios':[0.3,1],'height_ratios':[1,1,0.3],'hspace':0,'wspace':0,'bottom':0.05,'left':0.2,'right':0.95,'top':0.98})   
    for prop in ['Radius']: 
      ax[0,1].plot(np.log10(df['StellarMass'])[easy],(df[prop])[easy],'ko',label='Detectable')
      ax[0,1].plot(np.log10(df['StellarMass'])[~easy],(df[prop])[~easy],'ko',markerfacecolor='w',label='Undetectable')
      hist_x(np.log10(df['StellarMass']),easy,ax[2,1],color='k')
      hist_y(df[prop],easy,ax[0,0],color='k')
    
      bins_min=np.amin(np.log10(df['StellarMass']))
      bins_max=np.amax(np.log10(df['StellarMass']))
      ax[0,1].set_xlim(bins_min-0.05,bins_max+0.05)
      bins_min=np.amin((df[prop]))
      bins_max=np.amax((df[prop]))
      ax[0,1].set_ylim(bins_min-0.05,bins_max+0.05)
      ax[0,1].set_yticks([])
      ax[0,1].set_xticks([])
      ax[2,0].axis('off')
      ax[0,0].set_ylabel(r'$R_{0.5}/\rm{kpc}$')
    """
     
    OH_rad=np.zeros(len(df))
    for jj,ii in enumerate(df['Index']):
      if ii!=41:
        _pattern_OH= 'runJWST/SDSS_z7_SN_onlyHost/mcmc_out_mock_JWST_SDSS_{}_residual.fits'
        head_OH = fits.getheader(_pattern_OH.format(ii))
        OH_rad[jj]=head_OH['1SER_RE'].split(' ')[0]
      
    if filt in ['F277W','F356W','F444W']:
      pxscale = 0.063/2 #arcsec
    else:
      pxscale = 0.031/2
    
    OH_rad*=pxscale
    """
    ax[1,1].plot(np.log10(df['StellarMass'])[easy],OH_rad[easy],'ko',label='Detectable')
    ax[1,1].plot(np.log10(df['StellarMass'])[~easy],OH_rad[~easy],'ko',markerfacecolor='w',label='Undetectable')
    hist_y(OH_rad,easy,ax[1,0],color='k')
    
    bins_min=np.amin(np.log10(df['StellarMass']))
    bins_max=np.amax(np.log10(df['StellarMass']))
    ax[1,1].set_xlim(bins_min-0.05,bins_max+0.05)
    bins_min=np.amin((OH_rad))
    bins_max=np.amax((OH_rad))
    ax[1,1].set_ylim(bins_min-0.01,bins_max+0.01)
    ax[1,1].set_yticks([])
    ax[1,1].set_xticks([])
    """

    xdata=df['Radius']
    ydata=OH_rad
    
    fig,ax=plt.subplots(2,2,figsize=(4,3.3),gridspec_kw={'width_ratios':[0.3,1],'height_ratios':[1,0.3],'hspace':0,'wspace':0,'bottom':0.1,'left':0.2,'right':0.95,'top':0.95})   
    three_pane_plot(xdata,ydata,easy,ax)
    ax[1,1].set_xlabel(r'$R_{0.5}/\rm{kpc}$')
    ax[0,0].set_ylabel(r'F200W Sersic Radius (kpc)')
    ax[0,0].set_xlabel('Success\nRate')
    ax[0,1].legend(loc='lower right')
    plt.savefig('properties_radius.pdf') 
    #fig,ax=plt.subplots(2,3)
    
    #ax[1,0].plot(df['MUV_AGN'][easy]+df['tau_UV_AGN'][easy],df['MUV_gal_dust'][easy],'ko')
    #ax[1,0].plot(df['MUV_AGN'][~easy]+df['tau_UV_AGN'][~easy],df['MUV_gal_dust'][~easy],'ro')
    #ax[1,2].plot(df['MUV_AGN']+df['tau_UV_AGN']-df['MUV_gal_dust'],detectability,'ko')
    #plt.show()
    
    xdata=df['MUV_AGN_dust']
    ydata=df['MUV_gal_dust']
    
    fig,ax=plt.subplots(2,2,figsize=(4,3.3),gridspec_kw={'width_ratios':[0.3,1],'height_ratios':[1,0.3],'hspace':0,'wspace':0,'bottom':0.1,'left':0.2,'right':0.95,'top':0.95})   
    three_pane_plot(xdata,ydata,easy,ax)
    ax[1,1].set_xlabel(r'$M_{\rm{UV,~ AGN~ (dust)}}$')
    ax[0,0].set_ylabel(r'$M_{\rm{UV,~ host~ (dust)}}$')
    ax[0,0].set_xlabel('Success\nRate')
    plt.savefig('properties_magnitudes_dust.pdf') 
   
 
    xdata=df['MUV_AGN_dust']-1.086*df['tau_UV_AGN']
    ydata=df['MUV_gal']
    
    fig,ax=plt.subplots(2,2,figsize=(4,3.3),gridspec_kw={'width_ratios':[0.3,1],'height_ratios':[1,0.3],'hspace':0,'wspace':0,'bottom':0.1,'left':0.2,'right':0.95,'top':0.95})   
 
    three_pane_plot(xdata,ydata,easy,ax)
    ax[1,1].set_xlabel(r'$M_{\rm{UV,~ AGN}}$')
    ax[0,0].set_ylabel(r'$M_{\rm{UV,~ host}}$')
    ax[0,0].set_xlabel('Success\nRate')
    ax[0,1].legend(loc='lower right')
    plt.savefig('properties_magnitudes_intrinsic.pdf') 
   
    plt.show()

    """ 
    fig,ax=plt.subplots(2,4)   
    ii=0
    for prop in ['StellarMass','BHMass']:#'SFR    Radius
      success=np.zeros(6) #detectable in how many filters?
      data=np.log10(np.array(df[prop]))
      print(prop,data)
      bins_min=np.amin(data)
      bins_max=np.amax(data)
      bins_width=(bins_max-bins_min)/len(success)
      for ff,filt in enumerate(['F115W','F150W','F200W','F277W','F356W','F444W']):
        for bb in range(0,len(success)):
          b_min=bins_min+bb*bins_width
          b_max=bins_min+(bb+1)*bins_width
          success[bb]=np.sum(np.array(df[(data<b_max) & (data>b_min)][filt],dtype=int))/len(df[(data<b_max) & (data>b_min)])
        ax[0,ii].plot(bins_min+(np.arange(len(success))+0.5)*bins_width,success,'-',label=filt)
      ii+=1
        
    
    for prop in ['SFR','Radius','MUV_gal_dust']:
      success=np.zeros(6) #detectable in how many filters?
      data=np.array(df[prop])
      bins_min=np.amin(data)
      bins_max=np.amax(data)
      bins_width=(bins_max-bins_min)/len(success)
      for ff,filt in enumerate(['F115W','F150W','F200W','F277W','F356W','F444W']):
        for bb in range(0,len(success)):
          b_min=bins_min+bb*bins_width
          b_max=bins_min+(bb+1)*bins_width
          success[bb]=np.sum(np.array(df[(data<b_max) & (data>b_min)][filt],dtype=int))/len(df[(data<b_max) & (data>b_min)])
        if ii<4:
          ax[0,ii].plot(bins_min+(np.arange(len(success))+0.5)*bins_width,success,'-',label=filt)
        else:
          ax[1,ii-4].plot(bins_min+(np.arange(len(success))+0.5)*bins_width,success,'-',label=filt)
      ii+=1
    

    success=np.zeros(6) #detectable in how many filters?
    data=np.array(df['MUV_AGN']+df['tau_UV_AGN'])
    bins_min=np.amin(data)
    bins_max=np.amax(data)
    bins_width=(bins_max-bins_min)/len(success)
    for ff,filt in enumerate(['F115W','F150W','F200W','F277W','F356W','F444W']):
        for bb in range(0,len(success)):
          b_min=bins_min+bb*bins_width
          b_max=bins_min+(bb+1)*bins_width
          success[bb]=np.sum(np.array(df[(data<b_max) & (data>b_min)][filt],dtype=int))/len(df[(data<b_max) & (data>b_min)])
        ax[1,1].plot(bins_min+(np.arange(len(success))+0.5)*bins_width,success,'-',label=filt)
    
    success=np.zeros(6) #detectable in how many filters?
    data=np.array(df['MUV_AGN']+df['tau_UV_AGN']-df['MUV_gal_dust'])
    bins_min=np.amin(data)
    bins_max=np.amax(data)
    bins_width=(bins_max-bins_min)/len(success)
    for ff,filt in enumerate(['F115W','F150W','F200W','F277W','F356W','F444W']):
        for bb in range(0,len(success)):
          b_min=bins_min+bb*bins_width
          b_max=bins_min+(bb+1)*bins_width
          success[bb]=np.sum(np.array(df[(data<b_max) & (data>b_min)][filt],dtype=int))/len(df[(data<b_max) & (data>b_min)])
        ax[1,2].plot(bins_min+(np.arange(len(success))+0.5)*bins_width,success,'-',label=filt)

    plt.legend()
    plt.show()"""
