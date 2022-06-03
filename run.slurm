#!/bin/bash
# Do not forget to select a proper partition if
# the default one is no fit for the job!

#SBATCH --output=out.%j
#SBATCH --error=err.%j
#SBATCH --job-name=kes_10
#SBATCH --nodes=1          # number of nodes
#SBATCH --ntasks=2         # number of processor cores (i.e. tasks)
#SBATCH --gres=gpu:2 # number of tasks per node
#SBATCH --exclusive
#SBATCH --partition=long   # partition
#SBATCH --time=150:00:00    # walltime

# Good Idea to stop operation on first error.
set -e

# Load environment modules for your application here.
source /etc/profile.d/modules.sh
module load miniconda
module load cuda

# Actual work starting here.
# srun sleep 20 && hostname
srun python test1.py 
