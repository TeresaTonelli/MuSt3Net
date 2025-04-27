"""
Run script for the post-processing analysis: checking RMSE computation and float identification 
"""
import numpy as np
import torch 
import os 
import sys

sys.path.append("/leonardo_work/OGS23_PRACE_IT_0/ttonelli/CNN_reconstruction_final_resolution")


from float_identification import compute_list_float_devices, search_unseen_float, create_float_txt, select_float_measures, single_float_device_identifier, float_device_identifier
from hyperparameter import *
from rmse_functions import RMSE_ensemble_ga, RMSE_ensemble_season
from utils.utils_dataset_generation import read_list

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#RMSE computation
path_job = "/leonardo_work/OGS23_PRACE_IT_0/ttonelli/CNN_reconstruction_final_resolution/results_job_2025-04-18 16:23:24.817583"
years_week_indexes = read_list(path_job + "/results_training_2_ensemble/P_l/20/lrc_0.001" + "/ensemble_ywd_indexes.txt")
RMSE_ensemble_ga(path_job, years_week_indexes, 2)
#RMSE_ensemble_season(path_job, years_week_indexes, 2, threshold=20, behavior_season="bloom_DCM")

#Float devices identification
#list_float_devices = compute_list_float_devices("/leonardo_scratch/large/userexternal/ttonelli/OGS/SUPERFLOAT/SUPERFLOAT/Float_Index.txt")
#select_float_measures("/leonardo_scratch/large/userexternal/ttonelli/OGS/SUPERFLOAT/SUPERFLOAT/Float_Index.txt", '6901765/SD6901765_001.nc')
#list_unseen_floats, list_unseen_indexes = search_unseen_float("/leonardo_scratch/large/userexternal/ttonelli/OGS/SUPERFLOAT/SUPERFLOAT/Float_Index.txt", list_float_devices, [2019, 2020, 2021], 'CHLA')
#create_float_txt(list_unseen_indexes, "/leonardo_scratch/large/userexternal/ttonelli/OGS/SUPERFLOAT/SUPERFLOAT/Float_Index.txt")
#single_float_device_identifier("/leonardo_scratch/large/userexternal/ttonelli/OGS/SUPERFLOAT/SUPERFLOAT/Float_Index.txt", [2020, 6], [73, 271], [3.0, 3.0])
#float_device_identifier("/leonardo_scratch/large/userexternal/ttonelli/OGS/SUPERFLOAT/SUPERFLOAT/Float_Index.txt",[2021,4], [[62, 447], [139, 66]], [3.0, 3.0])