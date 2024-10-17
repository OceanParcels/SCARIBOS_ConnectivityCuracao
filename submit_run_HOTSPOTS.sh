#!/bin/bash
#SBATCH -J hotspots           # name of the job
#SBATCH -p normal             # partition, here 'normal' for jobs up to 120 hours
#SBATCH -n 4                  # number of cores
#SBATCH -t 5-00:00:00         # maximum run time, here 5 days
#SBATCH -o logfiles/log_run_HOTSPOTS.%j.out     # SLURM output file
#SBATCH -e logfiles/log_run_HOTSPOTS.%j.err     # SLURM error file

module load miniconda
eval "$(conda shell.bash hook)"                 # ensures conda works in the batch environment 

# Log start time
now="$(date)"
printf "Start date and time: %s\n" "$now"

# Navigate to the appropriate directory
cd /nethome/berto006/surface_run_parcels

# Activate the 'parcels' Conda environment
conda activate parcels

# Redirect Python terminal output to a file
s_time=$(date +%s)
for yr in {2020..2024}
do
    python 2_run_HOTSPOTS.py $yr > logfiles/log_run_HOTSPOTS_Y${yr}.o 2> logfiles/log_run_HOTSPOTS_Y${yr}.e &
done
wait
e_time=$(date +%s)
echo "Task completed Time: $(( e_time - s_time ))"
