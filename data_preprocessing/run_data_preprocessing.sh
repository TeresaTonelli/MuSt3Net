#!/usr/bin/env bash

#SBATCH --time=00:04:09

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --mem=128G
#SBATCH --partition=boost_usr_prod

#SBATCH --output=run_data_preprocessing.%j

#SBATCH --account=OGS23_PRACE_IT_0
#SBATCH --job-name=job_data_preprocessing

#SBATCH --cpus-per-task=1


module load python
source ~/.bashrc
module load anaconda3
conda activate ocean

python run_data_preprocessing.py