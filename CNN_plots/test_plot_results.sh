#!/usr/bin/env bash

#SBATCH --time=00:04:09

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --mem=64G
#SBATCH --partition=boost_usr_prod

#SBATCH --output=test_final_plots.%j

#SBATCH --account=OGS23_PRACE_IT_0
#SBATCH --job-name=job_test_plots

#SBATCH --cpus-per-task=1


module load python
source ~/.bashrc
module load anaconda3
conda activate ocean

python test_plot_results.py