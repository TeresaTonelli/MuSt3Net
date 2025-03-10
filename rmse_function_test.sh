#!/usr/bin/env bash

#SBATCH --time=00:34:29

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --mem=64G
#SBATCH --partition=boost_usr_prod

#SBATCH --output=rmse_computation.%j

#SBATCH --account=IscrC_MEDConNN
#SBATCH --job-name=job_rmse

#SBATCH --cpus-per-task=1


module load python
source ~/.bashrc
module load anaconda3
conda activate ocean

python rmse_function_test.py