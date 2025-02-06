#In this script we recreate the plots for POST-PORCESSING (so with already trained models) 


import numpy as np
import torch
import os

from convolutional_network import CompletionN
from normalization import Normalization
from utils_training_1 import load_tensors, load_land_sea_masks, load_old_total_tensor, load_transp_lat_coordinates
from plot_results_final import plot_NN_maps_final_1, plot_NN_maps_final_2, plot_models_profiles_1


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


input_tensor = load_tensors("dataset_training/total_dataset/P_l/2019/", [(24, 2)])
input_tensor, _, _ = Normalization(input_tensor, "1p", "results_job_2025-02-05 16:58:54.736919/results_training_1/mean_and_std_tensors")
input_tensor = input_tensor[0]
CNN_model = CompletionN()
path_job = "results_job_2025-02-05 16:58:54.736919"  
path_job_2 = "results_job_2025-01-28 08:55:13.496992"
list_masks = load_land_sea_masks("dataset_training/land_sea_masks/")
path_mean_std = "results_job_2025-02-05 16:58:54.736919/results_training_1/mean_and_std_tensors"
path_fig_channel = "results_job_2025-02-05 16:58:54.736919/results_training_1/P_l/200/lrc_0.001/plots"
path_fig_channel_prof = "results_job_2025-02-05 16:58:54.736919/results_training_1/P_l/200/lrc_0.001/plots_prof_trial"

tensor_output_num_model = load_old_total_tensor("dataset_training/old_total_dataset/", 0, [(2019, 24, 2)])
list_to_plot_coordinates = load_transp_lat_coordinates("dataset_training/total_dataset/P_l/2019/", [(24, 2)])

#plot_NN_maps_final_1(input_tensor, CNN_model, path_job, list_masks, "P_l", path_mean_std, path_fig_channel)
#plot_NN_maps_final_2(input_tensor, CNN_model, path_job_2, list_masks, "P_l", path_mean_std, path_fig_channel, 10)
plot_models_profiles_1(input_tensor, CNN_model, tensor_output_num_model, path_job, "P_l", path_mean_std, path_fig_channel_prof, list_to_plot_coordinates)