#In this script we test the rmse function, settings all the inputs that this function requires

import numpy as np
import torch 
import os 

from convolutional_network import CompletionN
from hyperparameter import *
from rmse_functions import select_season_tensors, create_ga_mask, compute_rmse_ga_season_2
from utils_function import compute_profile_coordinates
from utils_generation_train_1p import write_list, read_list
from utils_training_1 import load_land_sea_masks, re_load_tensors, re_load_old_float_tensors


path_job = "/leonardo_work/OGS23_PRACE_IT_0/ttonelli/CNN_reconstruction_final_resolution/results_job_2025-01-30 11:34:44.708444"
years_week_duplicates_indexes = read_list(path_job + "/results_training_1/P_l/200/lrc_0.001" + "/ywd_indexes.txt")[:100]
print("len years week indexes", len(years_week_duplicates_indexes), flush=True)
list_tensors = re_load_tensors("dataset_training/total_dataset", years_week_duplicates_indexes[:100])
print("len list tensors", len(list_tensors), flush=True)
season = "winter"
sst = select_season_tensors(list_tensors, season, years_week_duplicates_indexes)
print("sst", len(sst[0]), flush=True)

my_ga = "NWM"
list_float_tensors = re_load_old_float_tensors("/leonardo_work/OGS23_PRACE_IT_0/ttonelli/CNN_reconstruction_final_resolution/dataset", years_week_duplicates_indexes)
list_float_coordinates = [compute_profile_coordinates(torch.unsqueeze(float_tensor[:, -1, :, :, :], 1)) for float_tensor in list_float_tensors]
#print("single float coordinate", list_float_coordinates[1])
tensor_shape = (1, 1, d, h, w)
ga_masks = [create_ga_mask(my_ga, list_float_coordinates[i], tensor_shape) for i in range(len(list_float_coordinates))]
print("profiles counter", torch.count_nonzero(list_float_tensors[0]), flush=True)
print("ga mask counter", torch.count_nonzero(ga_masks[0]), flush=True)


land_sea_masks = load_land_sea_masks("dataset_training/land_sea_masks/")
exp_weights = torch.ones([1, 1, d-2, h, w-2])
CNN_model = CompletionN()
compute_rmse_ga_season_2(list_tensors, list_float_tensors, list_float_coordinates, land_sea_masks, my_ga, season, years_week_duplicates_indexes, exp_weights, CNN_model)