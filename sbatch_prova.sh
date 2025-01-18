#!/usr/bin/env bash


#SBATCH --time=00:01:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --mem=64G
#SBATCH --partition=boost_usr_prod

#SBATCH --output=hello_prova_env.%j

#SBATCH --account=OGS23_PRACE_IT_0
#SBATCH --job-name=hello_job

#SBATCH --cpus-per-task=1


module load python
source ~/.bashrc
module load anaconda3
conda activate ocean

python sbatch_prova.py
