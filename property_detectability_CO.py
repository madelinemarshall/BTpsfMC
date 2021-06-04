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
matplotlib.rcParams['font.size'] = (8)
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
    ax.set_ylim(-0.05,1.14)
    ax.set_yticks([0,1])
    ax.set_yticklabels(['0',''])
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
    ax.set_xlim(-0.05,1.14)
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

    BHAR_S=[14.500942,  13.576554,  10.321727,   8.939834,   8.855372,   8.808868,\
  8.62038,    8.601731,   7.5696898,  7.16324,    6.7052402,  6.6734014,\
  6.5346804,  6.1150928,  5.7489934,  5.2159667,  4.8983135,  4.826857,\
  4.396038,   4.2712193,  4.2201514,  2.0458283]
    EddRat_S=[2.08884382, 2.08884263, 2.08884468, 1.49570326, 1.73184892, 1.97549235,\
 2.08883628, 2.08884432, 2.08884368, 2.08883966, 1.45465988, 1.41140228,\
 1.28216163, 2.08883995, 0.72223113, 2.0304664,  2.08883499, 0.7652108,\
 0.75407579, 1.34089491, 0.93656116, 0.66053557]


    title='CO'
    indices=[4,5,11,13,14,15,19,26,28,31,34,35,37,38,42,44,49,51,52,53,54,56,57,61,62,63,64,67,68,71,77,78,82,85,86,87,94,97,98,101,103,105,106,110,111,112,113,114,119,124,125,126,128,129,130,133,138,143,145,147,148,149,151,156,158,159,161,163,165,172,173,177,178,179,181,182,184,186,189,191,195,196,199,200,202,204,205,206,207,214,215,218,219,221,222,223,226,231,233,234,235,241,244,246,248,249,251,252,255,258,259,261,265,266,268,269,270,272,274,275,278,281,290,291,292,297,298,302,304,306,308,311,315,317,319,324,328,335,337,345,346,348,353,363,366,367,369,380,384,388,390,396,409,410,427,429,437,442,459,460,461,464,465,477,483,487,505,519,528,546,553,561,572,606,612]
    
    df2=df[df['Sample']==title]
    detectable_CO=np.array(df2['F200W'],dtype='Bool')
    print(list(df2['Index']))
    print(df2['BHMass'])
    BHAR_CO=[11.940445,10.343307,8.602696,8.300257,7.988960,7.707414,7.130124,6.081528,5.609778,5.311627,5.171616,5.079066,4.845815,4.845482,4.404561,4.286726,4.055542,4.048377,3.906544,3.900786,3.804001,3.737399,3.694404,3.326599,3.240947,3.188533,3.155003,2.928020,2.909120,2.703755,2.494875,2.485669,2.394785,2.281305,2.232870,2.214507,2.142720,2.109287,2.101840,2.083977,2.038548,1.956089,1.929980,1.923307,1.854327,1.834500,1.766980,1.762516,1.743800,1.660430,1.611657,1.600296,1.567761,1.540797,1.538223,1.535284,1.508801,1.424180,1.350157,1.310436,1.269329,1.261723,1.258828,1.241989,1.191239,1.161009,1.157505,1.143901,1.125911,1.072921,1.021273,1.013291,0.964315,0.957282,0.950838,0.939281,0.938080,0.932379,0.928806,0.905721,0.902478,0.881879,0.870930,0.856811,0.848796,0.842527,0.841236,0.837810,0.836592,0.831998,0.798580,0.789102,0.773380,0.772240,0.753676,0.752329,0.742025,0.735712,0.716315,0.711220,0.708784,0.700511,0.687320,0.678819,0.677135,0.666157,0.664658,0.658379,0.656824,0.641578,0.636751,0.631319,0.627719,0.619564,0.615987,0.611784,0.611349,0.608631,0.603092,0.599612,0.597313,0.583809,0.580562,0.564681,0.558990,0.558418,0.545983,0.545849,0.540095,0.539203,0.529794,0.526727,0.520410,0.512253,0.510737,0.504455,0.497310,0.495358,0.488527,0.487708,0.486469,0.472988,0.472453,0.469870,0.462558,0.448302,0.444176,0.441372,0.438202,0.427428,0.423359,0.421268,0.420843,0.419588,0.410424,0.409332,0.393369,0.390724,0.381490,0.375128,0.363863,0.363763,0.362186,0.360785,0.360547,0.349483,0.346918,0.342693,0.335416,0.324439,0.318757,0.307143,0.304986,0.302745,0.298291,0.280871,0.279953]
    EddRat_CO=[2.088833,2.088841,2.088840,1.348868,2.087951,2.087940,2.088830,1.879501,1.708146,2.087945,2.087941,2.087944,0.588194,1.051335,0.730935,2.087943,2.088831,0.893843,0.663579,2.088843,2.087937,2.088837,2.087950,2.088848,1.410509,2.035651,1.094529,0.884196,2.087950,2.087937,2.088842,1.584401,2.088832,0.679503,1.689744,0.740116,0.569489,1.285330,1.062547,1.385341,2.087937,1.573261,1.669216,0.259793,0.679645,1.996959,0.685809,1.547203,0.458190,2.087935,0.507588,0.398051,0.348968,0.306726,0.614047,0.407684,0.633342,2.087946,1.298349,0.586384,0.871104,0.289284,0.240000,2.087951,1.478934,0.463450,0.347236,0.964607,1.556160,0.981461,2.087935,1.364010,0.215030,0.607570,0.348035,0.280113,0.379408,0.377928,0.228745,1.171679,0.597426,0.525938,2.087943,1.150091,2.087948,2.087948,0.210246,0.973818,0.942937,0.283522,1.432128,1.742890,0.246714,0.192936,0.476504,0.653258,0.722444,1.071522,0.458140,1.220530,0.791021,0.318125,0.223111,0.770166,0.986763,0.312942,0.151619,0.546866,0.292132,1.270897,2.087942,0.766596,0.282495,0.293413,0.130710,0.891647,0.515464,0.430878,0.151819,0.254761,0.462329,2.024968,0.309231,0.396943,0.674164,0.218427,0.388541,0.640879,0.243307,0.751615,0.228472,0.313613,0.633261,0.554603,1.405697,0.402837,0.129413,0.100977,0.208078,0.361036,0.633404,0.710612,0.211150,0.239859,0.336157,0.856349,0.345968,0.192773,1.006782,0.373830,0.419599,0.265646,0.431433,0.119572,0.192109,0.394372,0.346301,0.146120,1.872846,0.398922,0.228908,0.121235,0.202784,0.497996,0.167527,0.150138,0.150446,0.133245,0.019642,0.392422,0.227176,0.457542,0.149314,0.069437,0.269460,0.255158,0.094060]  
    BHAR_CO=np.delete(BHAR_CO,139)
    BHAR_CO=np.delete(BHAR_CO,37)
    EddRat_CO=np.delete(EddRat_CO,139)
    EddRat_CO=np.delete(EddRat_CO,37)

 
    if False: #Galaxy props
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
      ax [2,1].legend()
      plt.show()
      #plt.savefig('properties_stellarMass.pdf') 

    if True:#BHAR,Eddington ratio
    
      fig,ax=plt.subplots(3,2,figsize=(3.4,5.4),gridspec_kw={'width_ratios':[0.3,1],'height_ratios':[1,1,0.3],'hspace':0,'wspace':0,'bottom':0.06,'left':0.2,'right':0.95,'top':0.98})   
    
      xdata1=np.log10(df1['BHMass'])
      ydata1=np.array(BHAR_S)
      xdata2=np.log10(df2['BHMass'])
      ydata2=np.array(BHAR_CO)
      two_pane_plot(xdata1,ydata1,detectable_SDSS,xdata2,ydata2,detectable_CO,ax[0,1],ax[0,0])
 
      #ax[2,1].set_ylim(bins_min-0.05,bins_max+0.05)
      ax[0,0].set_ylabel(r'$\rm{BHAR}~ (M_\odot/\rm{yr})$')
 
      ydata1=np.array(EddRat_S)
      ydata2=np.array(EddRat_CO)
    
      two_pane_plot(xdata1,ydata1,detectable_SDSS,xdata2,ydata2,detectable_CO,ax[1,1],ax[1,0],bottom=ax[2,1])

      ax[2,1].set_xlabel(r'$\log(M_{\rm{BH}}/M_\odot)$')
      ax[1,0].set_ylabel(r'$L_{\rm{Bol}}/L_{\rm{Edd}}$')
      ax[1,0].set_xlabel('Success\nRate')
      ax[0,1].legend()
      ax[2,0].axis('off')
      #plt.savefig('properties_BHmassAccretion.pdf') 
      plt.show()
    
    
    fig,ax=plt.subplots(2,2,figsize=(3.4,2.8),gridspec_kw={'width_ratios':[0.3,1],'height_ratios':[1,0.3],'hspace':0,'wspace':0,'bottom':0.1,'left':0.2,'right':0.95,'top':0.95})   


    xdata1=df2['MUV_AGN_dust']-1.086*df2['tau_UV_AGN']
    ydata1=df2['MUV_gal']
    xdata2=df1['MUV_AGN_dust']-1.086*df1['tau_UV_AGN']
    ydata2=df1['MUV_gal']
    three_pane_plot(xdata1,ydata1,detectable_CO,xdata2,ydata2,detectable_SDSS,ax)
    ax[1,1].set_xlabel(r'$M_{\rm{UV,AGN~ (Intrinsic)}}$')
    ax[0,0].set_ylabel(r'$M_{\rm{UV,Host~ (Intrinsic)}}$')
    ax[0,0].set_xlabel('Success\nRate')
   
 
    
    fig,ax=plt.subplots(2,2,figsize=(3.2,3.2),gridspec_kw={'width_ratios':[0.3,1],'height_ratios':[1,0.3],'hspace':0,'wspace':0,'bottom':0.23,'left':0.2,'right':0.95,'top':0.95})   
 
    xdata1=df2['MUV_AGN_dust']
    ydata1=df2['MUV_gal_dust']
    xdata2=df1['MUV_AGN_dust']
    ydata2=df1['MUV_gal_dust']
    three_pane_plot(xdata1,ydata1,detectable_CO,xdata2,ydata2,detectable_SDSS,ax)
    
    ax[1,1].set_xlabel(r'$M_{\rm{UV,AGN}}$')
    ax[0,0].set_ylabel(r'$M_{\rm{UV,Host}}$')
    ax[0,0].set_xlabel('Success\nRate')
    ax[0,1].legend(fontsize='small',ncol=2,loc=(-0.45,-0.67))
    ax[0,0].invert_yaxis()
    ax[1,1].invert_xaxis()
    ax[0,1].invert_yaxis()
    ax[0,1].invert_xaxis()
    #plt.savefig('properties_magnitudes_intrinsic.pdf') 
    plt.savefig('properties_magnitudes_dust_CO.pdf') 
   
    plt.show()
