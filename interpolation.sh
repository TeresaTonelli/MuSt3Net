#!/usr/bin/env bash

#SBATCH --time=00:19:59

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --mem=64G
#SBATCH --partition=boost_usr_prod

#SBATCH --output=interpolation.%j

#SBATCH --account=OGS23_PRACE_IT_0
#SBATCH --job-name=job_interp

#SBATCH --cpus-per-task=1



module load python 
source ~/.bashrc
module load anaconda3
conda activate ocean

python interpolation.py