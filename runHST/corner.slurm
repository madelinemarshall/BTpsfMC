#!/bin/bash
#SBATCH --job-name=0005
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --time=0:20:00		#after this time the job will be killed
#SBATCH --mem-per-cpu=200	#memory in MB per core
#SBATCH -D /home/mmarshal/BLUETIDES/codes/BTpsfMC/runHST
#SBATCH -o corner.out		#output file
#SBATCH -e corner.err		#error file, will go to outfile if omitted

srun corner_plot save *_db.fits
