#!/bin/bash
# This script runs 2_run_COASTCON_ZONE1.py for all years and months
#SBATCH -J COASTCONrunZ1         # name of the job
#SBATCH -p normal             # for jobs upto 120 hours, there is also a short partition for jobs upto 3 hours
#SBATCH -n 12                 # number of cores
#SBATCH -t 5-00:00:00           # number of hours you want to reserve the cores
#SBATCH -o logfiles/log_run_COASTCON_zone1.%j.out     # name of the output file (=stuff you normally see on screen)
#SBATCH -e logfiles/log_run_COASTCON_zone1.%j.err     # name of the error file (if there are errors)

module load miniconda
eval "$(conda shell.bash hook)"  # this makes sure that conda works in the batch environment 
now="$(date)"
printf "Start date and time %s\n" "$now"
# once submitted to the batch tell the compute nodes where they should be
cd /nethome/berto006/surface_run_parcels

conda activate parcels

# Redirect Python terminal output to a file
s_time=$(date +%s)
for yr in {2020..2024}; do
    for mnt in {01..02}; do
        python 2_run_COASTCON_ZONE1.py $yr $mnt > logfiles/COASTCON/log_run_COASTCON_ZONE1_Y${yr}M${mnt}.o 2> logfiles/COASTCON/log_run_COASTCON_ZONE1_Y${yr}M${mnt}.e &
    done
    wait  # Wait for all months of a year to finish before proceeding to the next year
done
wait
e_time=$(date +%s)
echo "Task completed Time: $(( e_time - s_time ))"
