#!/bin/bash
# Example file to submit 2_run_REGIOCON_ARUB.py to the batch system
#SBATCH -J runARUBA         # name of the job
#SBATCH -p normal             # for jobs upto 120 hours, there is also a short partition for jobs upto 3 hours
#SBATCH -n 12                 # number of cores
#SBATCH -t 5-00:00:00           # number of hours you want to reserve the cores
#SBATCH -o logfiles/log_run_REGIOCON_ARUB.%j.out     # SLURM output file
#SBATCH -e logfiles/log_run_REGIOCON_ARUB.%j.err     # SLURM error file


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
    for mnt in {01..12}; do
        python 2_run_REGIOCON_ARUB.py $yr $mnt > logfiles/REGIOCON/log_run_REGIOCON_ARUB_Y${yr}M${mnt}.o 2> logfiles/REGIOCON/log_run_REGIOCON_ARUB_Y${yr}M${mnt}.e &
    done
    wait  # Wait for all months of a year to finish before proceeding to the next year
done
wait

e_time=$(date +%s)
total_time=$(( e_time - s_time ))

# Convert total time to hours, minutes, and seconds
hours=$(( total_time / 3600 ))
minutes=$(( (total_time % 3600) / 60 ))
seconds=$(( total_time % 60 ))

# Print end time and total task time
end_time="$(date)"
printf "End date and time: %s\n" "$end_time"
printf "Task completed. Total time: %d hours, %d minutes, %d seconds\n" "$hours" "$minutes" "$seconds"
