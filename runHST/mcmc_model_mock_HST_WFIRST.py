from numpy import array
from psfMC.ModelComponents import *
from psfMC.distributions import Normal, Uniform

zp = 25.9463

Configuration(obs_file='../data/sci_mock_HST_WFIRST_onlyHost.fits',
              #obsivm_file='../data/ivm_CFHQS-J0033-0125_H.fits.gz',             
              obsivm_file='../data/ivm_mock_HST_WFIRST_onlyHost.fits',
              psf_files='../data/sci_PSF_HST.fits',
              psfivm_files='../data/ivm_PSF_HST.fits',
              #psf_files='../data/sci_psf_00hr_H.fits.gz',
              #psfivm_files='../data/ivm_psf_00hr_H.fits.gz',
              #mask_file='../data/region.reg',
              mag_zeropoint=zp)

# We can treat the sky as an unknown component if the subtraction is uncertain
Sky(adu=0)#Normal(loc=0, scale=0.01))

center = array((35,35))

# Point source component
PointSource(xy=center,#Uniform(loc=array((96-0.5, 96-1)), scale=(2, 2)),
            mag=Uniform(loc=22, scale=6))

#Host galaxy
#xy_shift = array((4, 4))
#Sersic(xy=Uniform(loc=center-xy_shift, scale=2*xy_shift),
#      mag=Uniform(loc=22, scale=6),
#      reff=Uniform(loc=0.5, scale=10),
#      reff_b=Uniform(loc=0.5, scale=8),
#      index=Uniform(loc=0.5, scale=5),
#      angle=Uniform(loc=0, scale=180), angle_degrees=True)
