#!/bin/bash
#SBATCH --job-name=mockHST
#SBATCH --nodes=1
#SBATCH --tasks-per-node=16
#SBATCH --time=1:30:00		#after this time the job will be killed
#SBATCH --mem-per-cpu=200	#memory in MB per core
#SBATCH -D /home/mmarshal/BLUETIDES/codes/BTpsfMC/runHST
#SBATCH -o mcmc_HST.out		#output file
#SBATCH -e mcmc_HST.err		#error file, will go to outfile if omitted

source activate astroconda
python mcmc_mock_HST.py
