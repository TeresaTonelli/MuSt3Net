#!/usr/bin/env bash

#SBATCH --time=05:59:59

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --mem=256G  #200G
#SBATCH --partition=boost_usr_prod

#SBATCH --output=train_script_ensemble.%j

#SBATCH --account=OGS23_PRACE_IT_0
#SBATCH --job-name=job_ensemble_train_script

#SBATCH --cpus-per-task=1

#SBATCH --gres=gpu:1


module load python
source ~/.bashrc
module load anaconda3
conda activate ocean

python train_script_test_ensemble.py