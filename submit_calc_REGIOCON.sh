#!/bin/bash
# This script runs the 3_calc_REGIOCON.py script on 20 compute nodes (4 locations x 5 years)
#SBATCH -J calcREGIOCON           # name of the job
#SBATCH -p normal                 # for jobs upto 120 hours, there is also a short partition for jobs upto 3 hours
#SBATCH -n 20                     # number of cores
#SBATCH -t 5-00:00:00             # number of hours you want to reserve the cores
#SBATCH -o logfiles/log_calc_REGIOCON.%j.out     # name of the output file (=stuff you normally see on screen)
#SBATCH -e logfiles/log_calc_REGIOCON.%j.err     # name of the error file (if there are errors)

module load miniconda
eval "$(conda shell.bash hook)"  # this makes sure that conda works in the batch environment 
now="$(date)"
printf "Start date and time %s\n" "$now"
# once submitted to the batch tell the compute nodes where they should be
cd /nethome/berto006/surface_run_parcels

conda activate parcels

s_time=$(date +%s)

# loop over 4 locations (1, 2, 3, 4) and 5 years (2020, 2021, 2022, 2023, 2024)

for loc in 1 2 3 4
do
    for year in 2020 2021 2022 2023 2024
    do
        echo "Starting Location ${loc} and Year ${year}..."
        srun --exclusive --nodes=1 -n 1 --cpus-per-task=1 \
            python -u 3_calc_REGIOCON.py $loc $year \
            > logfiles/log_calc_REGIOCON_loc${loc}_year${year}.out \
            2> logfiles/log_calc_REGIOCON_loc${loc}_year${year}.err &
    done
done

# Wait for all background jobs to finish
wait

# Record the end time of the computation
e_time=$(date +%s)
echo "Task completed Time: $(( e_time - s_time )) seconds"