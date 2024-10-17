#!/bin/bash
# This script starts a single-domain CROCO computation on Linux

#SBATCH -J scaribV8                      # name of the job
#SBATCH -p normal                        # for jobs up to 120 hours, there is also a short partition for jobs up to 3 hours
#SBATCH -n 30                            # total number of cores
#SBATCH --nodes=1                        # number of nodes
#SBATCH --ntasks-per-node=30             # number of cores per node
#SBATCH -t 5-00:00:00                    # number of hours you want to reserve the cores
#SBATCH -o sbatch_SCARIBOSV8_part4.out   # name of the output file (=stuff you normally see on screen)
#SBATCH -e sbatch_SCARIBOSV8_part4.err   # name of the error file (if there are errors)

# Executable commands : 
umask 022
set -u

# once submitted to the batch tell the compute nodes where they should be
cd /nethome/berto006/croco/CONFIG/SCARIBOS_V8

# launch the run script
./run_croco_inter.bash

