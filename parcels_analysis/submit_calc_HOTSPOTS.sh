#!/bin/bash
#SBATCH -J calcHOTSPOTS             # name of the job
#SBATCH -p normal               # for jobs upto 120 hours, there is also a short partition for jobs upto 3 hours
#SBATCH -n 1                    # number of cores
#SBATCH -t 1-08:00:00           # number of hours you want to reserve the cores
#SBATCH -o logfiles/log_calc_HOTSPOTS.out     # name of the output file (=stuff you normally see on screen)
#SBATCH -e logfiles/log_calc_HOTSPOTS.err     # name of the error file (if there are errors)

module load miniconda
eval "$(conda shell.bash hook)"  # this makes sure that conda works in the batch environment 

# once submitted to the batch tell the compute nodes where they should be
cd /nethome/berto006/surface_run_parcels/

conda activate parcels

# Redirect Python terminal output to a file
python -u 3_calc_HOTSPOTS.py > logfiles/log_calc_HOTSPOTS.log 2>&1

