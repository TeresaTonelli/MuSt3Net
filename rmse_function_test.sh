#!/usr/bin/env bash

#SBATCH --time=00:01:59

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --mem=300G
#SBATCH --partition=boost_usr_prod

#SBATCH --output=rmse_computation.%j

#SBATCH --account=OGS23_PRACE_IT_0
#SBATCH --job-name=job_rmse

#SBATCH --cpus-per-task=1


module load python
source ~/.bashrc
module load anaconda3
conda activate ocean

python rmse_function_test.py