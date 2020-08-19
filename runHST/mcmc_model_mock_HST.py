from numpy import array
from psfMC.ModelComponents import *
from psfMC.distributions import Normal, Uniform

zp = 25.9463

Configuration(obs_file='../data/sci_mock_HST.fits',
              obsivm_file='../data/ivm_mock_HST.fits',
              psf_files='../data/sci_PSF_HST.fits',
              psfivm_files='../data/ivm_PSF_HST.fits',
              #psf_files='../data/sci_psf_00hr_H.fits.gz',
              #psfivm_files='../data/ivm_psf_00hr_H.fits.gz',
              #mask_file='../data/region.reg',
              mag_zeropoint=zp)

# We can treat the sky as an unknown component if the subtraction is uncertain
Sky(adu=Normal(loc=0, scale=0.01))

# Point source component
PointSource(xy=Uniform(loc=array((64-2, 64-2)), scale=(4.0, 4.0)),
            mag=Uniform(loc=20, scale=4))
