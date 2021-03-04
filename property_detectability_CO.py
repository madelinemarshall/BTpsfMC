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

def hist_x(data,mask,ax,color='#d6f1ff'):
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
          #err[bb]=np.sqrt(len(data[mask][(data[mask]<=b_max) & (data[mask]>b_min)]))/len(data[(data<=b_max) & (data>b_min)])
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
    ax.bar(bins_min+bins_width*(0.5+np.arange(0,nbins)),height=frac,width=bins_width,edgecolor='k',facecolor=color,yerr=[err,err_up],capsize=3,ecolor=[0.5,0.5,0.5])#[0.8,0.8,0.8])
    #ax.plot(bins_min+bins_width*(1/2+np.arange(0,6)),frac,'ro',zorder=99)#, linewidth=1.2,marker='o',**kwargs)
    #ax.set_xlim(bins_min-0.05,bins_max+0.05)
    #ax.set_ylim(-0.05,1.171)
    ax.set_yticks([0,1])
    ax.tick_params(axis='both', direction='in')
    return

def hist_y(data,mask,ax,color='#d6f1ff'):
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
    ax.barh(bins_min+bins_width*(0.5+np.arange(0,nbins)),width=frac,height=bins_width,edgecolor='k',facecolor=color,xerr=[err,err_up],capsize=3,ecolor=[0.5,0.5,0.5])#[0.8,0.8,0.8])
    #ax.plot(frac,bins_min+bins_width*(1/2+np.arange(0,6)),'ro',zorder=99)#, linewidth=1.2,marker='o',**kwargs)
    #ax.set_ylim(bins_min-0.05,bins_max+0.05)
    #ax.set_xlim(-0.05,1.14)
    ax.set_xticks([0,1])
    ax.tick_params(axis='both', direction='in')
    return

def two_pane_plot(xdata1,ydata1,mask1,xdata2,ydata2,mask2,ax,hist_ax,bottom=False):
    ax.plot(xdata2[mask2],ydata2[mask2],'o',label='Detectable',color='#533ba1')
    ax.plot(xdata2[~mask2],ydata2[~mask2],'o',markerfacecolor='w',label='Undetectable',color='#533ba1')
    ax.plot(xdata1[mask1],ydata1[mask1],'ko',label='Detectable')
    ax.plot(xdata1[~mask1],ydata1[~mask1],'ko',markerfacecolor='w',label='Undetectable')

    xdata=np.append(xdata1,xdata2)
    ydata=np.append(ydata1,ydata2)
    mask=np.append(mask1,mask2)

    if bottom:
      hist_x(xdata,mask,bottom)
    hist_y(ydata,mask,hist_ax)
    
    bins_min=np.amin(xdata)
    bins_max=np.amax(xdata)
    ax.set_xlim(bins_min-0.05,bins_max+0.05)
    if bottom:
      bottom.set_xlim(bins_min-0.05,bins_max+0.05)
    bins_min=np.amin(ydata)
    bins_max=np.amax(ydata)
    ax.set_ylim(bins_min-0.05,bins_max+0.05)
    hist_ax.set_ylim(bins_min-0.05,bins_max+0.05)
    ax.set_yticks([])
    ax.set_xticks([])
    return


def three_pane_plot(xdata1,ydata1,mask1,xdata2,ydata2,mask2,ax):
    ax[0,1].plot(xdata1[mask1],ydata1[mask1],'o',label='Detectable in F200W',color='#84bda8')#,color='#984ea3')         #['#e41a1c','#377eb8','#4daf4a','#984ea3')
    ax[0,1].plot(xdata1[~mask1],ydata1[~mask1],'o',markerfacecolor='w',label='Undetectable in F200W',color='#84bda8')#,color='#984ea3')
    ax[0,1].plot([0,0],[0,0],'o',label='Faint Quasars',color='#84bda8')
    ax[0,1].plot(xdata2[mask2],ydata2[mask2],'ko',label='__nolabel__')#,color=color)#,color='#984ea3')         #['#e41a1c','#377eb8','#4daf4a','#984ea3')
    ax[0,1].plot(xdata2[~mask2],ydata2[~mask2],'ko',markerfacecolor='w',label='__nolabel__')#,color=color)#,color='#984ea3')
    ax[0,1].plot([0,0],[0,0],'ko',label='Bright Quasars')

    xdata=np.append(xdata1,xdata2)
    ydata=np.append(ydata1,ydata2)
    mask=np.append(mask1,mask2)

    hist_x(xdata,mask,ax[1,1],color='#c7eddf')
    hist_y(ydata,mask,ax[0,0],color='#c7eddf')
    
    bins_min=np.amin(xdata)
    bins_max=np.amax(xdata)
    ax[0,1].set_xlim(bins_min-0.1,bins_max+0.1)
    ax[1,1].set_xlim(bins_min-0.1,bins_max+0.1)
    bins_min=np.amin(ydata)
    bins_max=np.amax(ydata)
    ax[0,0].set_ylim(bins_min-0.1,bins_max+0.1)
    ax[0,1].set_ylim(bins_min-0.1,bins_max+0.1)
    ax[0,1].set_yticks([])
    ax[0,1].set_xticks([])
    ax[1,0].axis('off')
    return

if __name__=='__main__':
   
    title='SDSS'
    indices = [2,   3,   6,   7,   8,   9,  10,  12,  16,  18,  20,  22,  23,  25,  27,  32,  36,  40,   43,  45,  46, 100]
    
    df=pd.read_pickle('/home/mmarshal/BLUETIDES/BlueTides/PIG_208/processed_data/quasarDatabase_processed.pkl')
    #df.rename(columns={"MUV_AGN":"MUV_AGN_dust"}, inplace = True)
    #df.to_pickle('/home/mmarshal/BLUETIDES/BlueTides/PIG_208/processed_data/quasarDatabase_processed.pkl')
    print(df.columns)


    df1=df[df['Sample']==title]
    detectable_SDSS=np.array(df1['F200W'],dtype='Bool')

    title='CO'
    indices=[4,5,11,13,14,15,19,26,28,31,34,35,37,38,42,44,49,51,52,53,54,56,57,61,62,63,64,67,68,71,77,78,82,85,86,87,94,97,98,101,103,105,106,110,111,112,113,114,119,124,125,126,128,129,130,133,138,143,145,147,148,149,151,156,158,159,161,163,165,172,173,177,178,179,181,182,184,186,189,191,195,196,199,200,202,204,205,206,207,214,215,218,219,221,222,223,226,231,233,234,235,241,244,246,248,249,251,252,255,258,259,261,265,266,268,269,270,272,274,275,278,281,290,291,292,297,298,302,304,306,308,311,315,317,319,324,328,335,337,345,346,348,353,363,366,367,369,380,384,388,390,396,409,410,427,429,437,442,459,460,461,464,465,477,483,487,505,519,528,546,553,561,572,606,612]
    
    df2=df[df['Sample']==title]
    detectable_CO=np.array(df2['F200W'],dtype='Bool')
   
 
    fig,ax=plt.subplots(4,2,figsize=(4,8),gridspec_kw={'width_ratios':[0.3,1],'height_ratios':[1,1,1,0.3],'hspace':0,'wspace':0,'bottom':0.05,'left':0.2,'right':0.95,'top':0.95})   
    
    labs={'BHMass':r'$\log(M_{\rm{BH}}/M_\odot)$','SFR':r'$\log(\rm{SFR}/M_\odot \rm{yr}^{-1})$'}
    for ii,prop in enumerate(['BHMass','SFR']): 
      #fig,ax=plt.subplots(1,2,figsize=(4,3),gridspec_kw={'width_ratios':[0.3,1],'wspace':0,'bottom':0.2})   
      xdata1=np.log10(df1['StellarMass'])
      ydata1=np.log10(df1[prop])
      xdata2=np.log10(df2['StellarMass'])
      ydata2=np.log10(df2[prop])
      two_pane_plot(xdata1,ydata1,detectable_SDSS,xdata2,ydata2,detectable_CO,ax[ii,1],ax[ii,0])
      ax[ii,0].set_ylabel(labs[prop])
      #ax[ii,1].set_xlabel(r'$\log(M_\ast/M_\odot)$')
    
    xdata1=np.log10(df1['StellarMass'])
    ydata1=df1['Radius']
    xdata2=np.log10(df2['StellarMass'])
    ydata2=df2['Radius']
    two_pane_plot(xdata1,ydata1,detectable_SDSS,xdata2,ydata2,detectable_CO,ax[2,1],ax[2,0],bottom=ax[3,1])
    ax[3,0].axis('off')
    ax[2,0].set_ylabel(r'$R_{0.5}/\rm{kpc}$')
    ax[3,1].set_xlabel(r'$\log(M_\ast/M_\odot)$')
    ax[2,0].set_xlabel('Success\nRate')
    ax[2,1].legend()
    plt.show()
    #plt.savefig('properties_stellarMass.pdf') 
   
    #fig,ax=plt.subplots(2,3)
   
    #ax[1,0].plot(df['MUV_AGN'][detectable_SDSS]+df['tau_UV_AGN'][detectable_SDSS],df['MUV_gal_dust'][detectable_SDSS],'ko')
    #ax[1,0].plot(df['MUV_AGN'][~detectable_SDSS]+df['tau_UV_AGN'][~detectable_SDSS],df['MUV_gal_dust'][~detectable_SDSS],'ro')
    #ax[1,2].plot(df['MUV_AGN']+df['tau_UV_AGN']-df['MUV_gal_dust'],detectability,'ko')
    #plt.show()
    
    
    fig,ax=plt.subplots(2,2,figsize=(4,3.3),gridspec_kw={'width_ratios':[0.3,1],'height_ratios':[1,0.3],'hspace':0,'wspace':0,'bottom':0.1,'left':0.2,'right':0.95,'top':0.95})   


    xdata1=df2['MUV_AGN_dust']-1.086*df2['tau_UV_AGN']
    ydata1=df2['MUV_gal']
    xdata2=df1['MUV_AGN_dust']-1.086*df1['tau_UV_AGN']
    ydata2=df1['MUV_gal']
    three_pane_plot(xdata1,ydata1,detectable_CO,xdata2,ydata2,detectable_SDSS,ax)
    ax[1,1].set_xlabel(r'$M_{\rm{UV,AGN~ (Intrinsic)}}$')
    ax[0,0].set_ylabel(r'$M_{\rm{UV,Host~ (Intrinsic)}}$')
    ax[0,0].set_xlabel('Success\nRate')
   
 
    
    fig,ax=plt.subplots(2,2,figsize=(4,3.9),gridspec_kw={'width_ratios':[0.3,1],'height_ratios':[1,0.3],'hspace':0,'wspace':0,'bottom':0.23,'left':0.2,'right':0.95,'top':0.95})   
 
    xdata1=df2['MUV_AGN_dust']
    ydata1=df2['MUV_gal_dust']
    xdata2=df1['MUV_AGN_dust']
    ydata2=df1['MUV_gal_dust']
    three_pane_plot(xdata1,ydata1,detectable_CO,xdata2,ydata2,detectable_SDSS,ax)
    
    ax[1,1].set_xlabel(r'$M_{\rm{UV,AGN}}$')
    ax[0,0].set_ylabel(r'$M_{\rm{UV,Host}}$')
    ax[0,0].set_xlabel('Success\nRate')
    ax[0,1].legend(fontsize='small',ncol=2,loc=(-0.27,-0.65))
    ax[0,0].invert_yaxis()
    ax[1,1].invert_xaxis()
    ax[0,1].invert_yaxis()
    ax[0,1].invert_xaxis()
    #plt.savefig('properties_magnitudes_intrinsic.pdf') 
    plt.savefig('properties_magnitudes_dust_CO.pdf') 
   
    plt.show()
