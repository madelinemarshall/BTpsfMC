from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#plt.rc('text', usetex=True)
plt.rc('font', family='serif')
colors         = ['#e41a1c','#377eb8','#4daf4a','#984ea3',\
                  '#ff9f05','#ff7300','#ffbc05']#ffb405
                  #'#ff7f00','#ffe02e','#f781bf','#98ff98']*4

def maxL(filename):
  hdul = fits.open(filename+'_db.fits')
  lnp=hdul[1].data['lnprobability']
  return np.max(lnp)

def read_n(filename):
  hdul = fits.open(filename+'_residual.fits')
  return hdul[0].header['NAXIS1']

def calc_BIC(k,n,maxl):
  return k*np.log(n*n)-2*maxl


def diff_BIC(filename_host,filename_noHost):
  nn = read_n(filename_host)  
  BIC_host = calc_BIC(8,nn,maxL(filename_host))
  BIC_noHost = calc_BIC(1,nn,maxL(filename_noHost))
  if BIC_host<BIC_noHost-10:
    #print('Host model preferred, BIC = {}'.format(BIC_noHost-BIC_host))
    return True
  else:
    #print('No host model preferred, BIC = {}'.format(BIC_noHost-BIC_host))
    return False

def assess_quasar(folder_noHost,folder_host,fname,quasars):
  detection=np.zeros(len(quasars),dtype='Bool')
  for ii,q_ind in enumerate(quasars):
    filename_noHost=folder_noHost+fname+str(q_ind)
    filename_host=folder_host+fname+str(q_ind)

    diff_BIC(filename_host,filename_noHost)
    detection[ii]=diff_BIC(filename_host,filename_noHost)
  print('Success for {}/{}'.format(len(detection[detection]),len(detection)))
  print(np.array(quasars)[detection==False])
  return detection,(len(detection[detection])/len(detection))


if __name__=='__main__':
  quasars = [2,   3,   6,   7,   8,   9, 10,  12,  16,  18,  20, 22,  23,  25,  27,  32, 36,  40,   43,  45,  46, 100]
  success = {}
  detect_rate = {}

  #property dataframe
  df=pd.read_pickle('/home/mmarshal/BLUETIDES/BlueTides/PIG_208/processed_data/quasarDatabase.pkl')
  df=df[df.Index!=41]#quasar 41s spectra doesn't work
  title='SDSS'
  
  #HST
  print('HST')
  folder_noHost = '/home/mmarshal/BLUETIDES/codes/BTpsfMC/runHST/SDSS_z7_noHost/'
  folder_host = '/home/mmarshal/BLUETIDES/codes/BTpsfMC/runHST/SDSS_z7/'
  fname='mcmc_out_mock_HST_SDSS_'
  success['HST'],detect_rate['HST']=assess_quasar(folder_noHost,folder_host,fname,quasars)
  
  #JWST
  print('JWST F150W 4800s')
  folder_noHost = '/home/mmarshal/BLUETIDES/codes/BTpsfMC/runJWST/SDSS_z7_F150W_4800s_noHost/'
  folder_host = '/home/mmarshal/BLUETIDES/codes/BTpsfMC/runJWST/SDSS_z7_F150W_4800s/'
  fname='mcmc_out_mock_JWST_SDSS_'
  success['F150W 4800s'],detect_rate['F150W 4800s']=assess_quasar(folder_noHost,folder_host,fname,quasars)
  
  #JWST
  print('____Exposure Times____')
  print('1000s')
  folder_noHost = '/home/mmarshal/BLUETIDES/codes/BTpsfMC/runJWST/SDSS_z7_1000s_noHost/'
  folder_host = '/home/mmarshal/BLUETIDES/codes/BTpsfMC/runJWST/SDSS_z7_1000s/'
  success['1 ks'],detect_rate['1 ks']=assess_quasar(folder_noHost,folder_host,fname,quasars)
  
  print('2500s')
  folder_noHost = '/home/mmarshal/BLUETIDES/codes/BTpsfMC/runJWST/SDSS_z7_2500s_noHost/'
  folder_host = '/home/mmarshal/BLUETIDES/codes/BTpsfMC/runJWST/SDSS_z7_2500s/'
  success['2.5 ks'],detect_rate['2.5 ks']=assess_quasar(folder_noHost,folder_host,fname,quasars)
  
  print('5000s')
  folder_noHost = '/home/mmarshal/BLUETIDES/codes/BTpsfMC/runJWST/SDSS_z7_5000s_noHost/'
  folder_host = '/home/mmarshal/BLUETIDES/codes/BTpsfMC/runJWST/SDSS_z7_5000s/'
  success['5 ks'],detect_rate['5 ks']=assess_quasar(folder_noHost,folder_host,fname,quasars)

  print('10000s')
  folder_noHost = '/home/mmarshal/BLUETIDES/codes/BTpsfMC/runJWST/SDSS_z7_SN_noHost/'
  folder_host = '/home/mmarshal/BLUETIDES/codes/BTpsfMC/runJWST/SDSS_z7_SN/'
  success['F200W'],detect_rate['F200W']=assess_quasar(folder_noHost,folder_host,fname,quasars)
  wavelength={'HST':1.6,'F150W 4800s':1.5,'1 ks':2.0,'5 ks':2.0,'F200W':2.0}

  for filt in ['F115W','F150W','F277W','F356W','F444W','F560W']:
    print(filt)
    folder_noHost = '/home/mmarshal/BLUETIDES/codes/BTpsfMC/runJWST/SDSS_z7_'+filt+'_noHost/'
    folder_host = '/home/mmarshal/BLUETIDES/codes/BTpsfMC/runJWST/SDSS_z7_'+filt+'/'
    success[filt],detect_rate[filt]=assess_quasar(folder_noHost,folder_host,fname,quasars)
  wavelength2={'F115W':1.15,'F150W':1.5,'F277W':2.77,'F356W':3.56,'F444W':4.44,'F560W':5.60}
  wavelength.update(wavelength2)
  
  print('F277W 5000s')
  folder_noHost = '/home/mmarshal/BLUETIDES/codes/BTpsfMC/runJWST/SDSS_z7_F277W_5000s_noHost/'
  folder_host = '/home/mmarshal/BLUETIDES/codes/BTpsfMC/runJWST/SDSS_z7_F277W_5000s/'
  success['F277W 5 ks'],detect_rate['F277W 5 ks']=assess_quasar(folder_noHost,folder_host,fname,quasars)

  
  for key in success.keys():
    df.loc[(df['Sample']==title),key]=success[key]    
  print(df.loc[df['Sample']==title])
  df.to_pickle('/home/mmarshal/BLUETIDES/BlueTides/PIG_208/processed_data/quasarDatabase_processed.pkl')
  
  """print('CO quasars')
  title='CO'
  quasars=[4,5,11,13,14,15,19,26,28,31,34,35,37,38,42,44,49,51,52,53,54,56,57,61,62,63,64,67,68,71,77,78,82,85,86,87,94,97,98,101,103,105,106,110,111,112,113,114,119,124,125,126,128,129,130,133,138,143,145,147,148,149,151,156,158,159,161,163,165,172,173,177,178,179,181,182,184,186,189,191,195,196,199,200,202,204,205,206,207,214,215,218,219,221,222,223,226,231,233,234,235,241,244,246,248,249,251,252,255,258,259,261,265,266,268,269,270,272,274,275,278,281,290,291,292,297,298,302,304,306,308,311,315,317,319,324,328,335,337,345,346,348,353,363,366,367,369,380,384,388,390,396,409,410,427,429,437,442,459,460,461,464,465,477,483,487,505,519,528,546,553,561,572,606,612]
  folder_noHost = '/home/mmarshal/BLUETIDES/codes/BTpsfMC/runJWST/CO_z7_noHost/'
  folder_host = '/home/mmarshal/BLUETIDES/codes/BTpsfMC/runJWST/CO_z7/'
  fname='mcmc_out_mock_JWST_CO_'
  a,b=assess_quasar(folder_noHost,folder_host,fname,quasars)"""

  #Plot success rates
  fig = plt.subplots(figsize=(4,4),gridspec_kw={'bottom':0.3,'left':0.15,'right':0.95})
  
  plt.plot(wavelength['1 ks'],detect_rate['1 ks'],'o',color=colors[6],label='1 ks')
  plt.plot(2.0,detect_rate['2.5 ks'],'o',color=colors[4],label='2.5 ks')
  plt.plot(wavelength['5 ks'],detect_rate['5 ks'],'o',color=colors[5],label='5 ks')

  for filt in ['F115W','F150W','F277W','F356W','F444W']:
    plt.plot(wavelength[filt],detect_rate[filt],'o',color=colors[0],label='__nolabel__')#filt)
  plt.plot(wavelength['F200W'],detect_rate['F200W'],'o',color=colors[0],label='10 ks')
  
  plt.plot(wavelength['F150W 4800s'],detect_rate['F150W 4800s'],'o',color=colors[1],label='4.8 ks')
  plt.plot(wavelength['HST'],detect_rate['HST'],'s',color=colors[1],label='4.8 ks (HST WFC3)')
  plt.plot(wavelength['F560W'],detect_rate['F560W'],'^',color=colors[0],label='10 ks (MIRI)')
  plt.plot(wavelength['F277W'],detect_rate['F277W 5 ks'],'o',color=colors[5],label='__nolabel__')#'5 ks')

  
  plt.legend(fontsize='small',loc=(0.1,-0.5),ncol=2)
  plt.xlabel('Wavelength (microns)')
  plt.ylabel('Fraction of Successful Detections')
  #plt.tight_layout()
  plt.savefig('success_rates.pdf')
  plt.show()


 
