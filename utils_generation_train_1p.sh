#!/usr/bin/env bash

#SBATCH --time=00:02:59

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --mem=256G  #200G
#SBATCH --partition=boost_usr_prod

#SBATCH --output=gen_data_train_1p.%j

#SBATCH --account=OGS23_PRACE_IT_0
#SBATCH --job-name=job_data_gen_train_1p

#SBATCH --cpus-per-task=1


module load python
source ~/.bashrc
module load anaconda3
conda activate ocean

python utils_generation_train_1p.py