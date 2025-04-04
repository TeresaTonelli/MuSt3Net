#!/usr/bin/env bash

#SBATCH --time=00:00:09

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --mem=16G
#SBATCH --partition=boost_usr_prod

#SBATCH --output=biogeoch_float_data_gen.%j

#SBATCH --account=IscrC_MEDConNN
#SBATCH --job-name=job_float_biogeoch

#SBATCH --cpus-per-task=1



module load python
source ~/.bashrc
module load anaconda3
conda activate ocean

python make_dataset_biogeochemical_float.py
