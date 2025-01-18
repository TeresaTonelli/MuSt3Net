#!/usr/bin/env bash

#SBATCH --time=00:00:20

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --mem=64G
#SBATCH --partition=boost_usr_prod

#SBATCH --output=plot_st.%j

#SBATCH --account=OGS23_PRACE_IT_0
#SBATCH --job-name=job_plot_st

#SBATCH --cpus-per-task=1



module load python 
source ~/.bashrc
module load anaconda3
conda activate ocean

python plot_save_tensor.py
