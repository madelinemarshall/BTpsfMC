from numpy import array
from psfMC.ModelComponents import *
from psfMC.distributions import Normal, Uniform

zp = 25.9463

Configuration(obs_file='../data/sci_mock_JWST.fits',
              obsivm_file='../data/ivm_mock_JWST.fits',
              psf_files='../data/sci_PSF_JWST.fits',
              psfivm_files='../data/ivm_PSF_JWST.fits',
              mag_zeropoint=zp)

Sky(adu=0)

PointSource(xy=[96,96],
            mag=Uniform(loc=20, scale=6))
