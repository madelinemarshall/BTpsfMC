#!/bin/bash
#SBATCH --job-name=mockJWST
#SBATCH --nodes=1
#SBATCH --tasks-per-node=16
#SBATCH --time=1:30:00		#after this time the job will be killed
#SBATCH --mem-per-cpu=200	#memory in MB per core
#SBATCH -D /home/mmarshal/BLUETIDES/codes/BTpsfMC/runJWST
#SBATCH -o mcmc_mock_JWST.out		#output file
#SBATCH -e mcmc_mock_JWST.err		#error file, will go to outfile if omitted

source activate astroconda
python mcmc_mock_JWST.py
