#!/bin/bash
# This script runs the 3_calc_COASTCON.py script on X compute nodes (X = number of zones)
#SBATCH -J calcCOASTCON           # name of the job
#SBATCH -p normal             # for jobs upto 120 hours, there is also a short partition for jobs upto 3 hours
#SBATCH -n 8                  # number of cores
#SBATCH -t 5-00:00:00         # number of hours you want to reserve the cores
#SBATCH -o logfiles/calc_COASTCON.%j.out     # name of the output file (=stuff you normally see on screen)
#SBATCH -e logfiles/calc_COASTCON.%j.err     # name of the error file (if there are errors)

module load miniconda
eval "$(conda shell.bash hook)"  # this makes sure that conda works in the batch environment 
now="$(date)"
printf "Start date and time %s\n" "$now"
# once submitted to the batch tell the compute nodes where they should be
cd /nethome/berto006/surface_run_parcels

conda activate parcels

# Redirect Python terminal output to a file
s_time=$(date +%s)

# Loop through zones and submit each to a single core
for zone in {1..8}; do
    echo "Starting Zone ${zone}..."
    
    # Use srun to run each zone on 1 core exclusively
    srun --exclusive -n 1 --cpus-per-task=1 \
        python -u 3_calc_COASTCON.py $zone \
        > logfiles/log_calc_COASTCON_zone${zone}.out \
        2> logfiles/log_calc_COASTCON_zone${zone}.err &
done

# Wait for all background jobs to finish
wait

# Record the end time of the computation
e_time=$(date +%s)
echo "Task completed Time: $(( e_time - s_time )) seconds"