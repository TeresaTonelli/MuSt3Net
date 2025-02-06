#!/usr/bin/env bash

#SBATCH --time=00:00:59

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --mem=256G  #200G
#SBATCH --partition=boost_usr_prod

#SBATCH --output=random_gen.%j

#SBATCH --account=OGS23_PRACE_IT_0
#SBATCH --job-name=job_random

#SBATCH --cpus-per-task=1


module load python
source ~/.bashrc
module load anaconda3
conda activate ocean

python data_generation_script.py