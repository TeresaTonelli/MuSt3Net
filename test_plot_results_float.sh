#!/usr/bin/env bash

#SBATCH --time=00:00:19

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --mem=4G
#SBATCH --partition=boost_usr_prod

#SBATCH --output=test_float_plots.%j

#SBATCH --account=IscrC_MEDConNN
#SBATCH --job-name=job_test_plots

#SBATCH --cpus-per-task=1


module load python
source ~/.bashrc
module load anaconda3
conda activate ocean

python test_plot_results_float.py