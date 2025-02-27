#In this script we recreate the plots for POST-PORCESSING (so with already trained models) 


import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns

from convolutional_network import CompletionN
from hyperparameter import *
from normalization import Normalization, tmp_Normalization
from utils_training_1 import load_tensors, load_land_sea_masks, load_old_total_tensor, load_transp_lat_coordinates
from plot_results_final import plot_NN_maps_final_1, plot_NN_maps_final_2, plot_models_profiles_1, plot_models_profiles_2
from plot_results import plot_NN_maps


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

sns.set_theme(context='paper', style='whitegrid', font='sans-serif', font_scale=1.5,
              color_codes=True, rc=None)

##path_job = "/leonardo_work/OGS23_PRACE_IT_0/ttonelli/CNN_reconstruction_final_resolution/results_job_2025-02-22 10:42:50.013434"
##path_results_1 = path_job + "/results_training_1"
##input_tensor = load_tensors("dataset_training/total_dataset/P_l/2019/", [(24, 2)])
##input_tensor, _, _ = Normalization(input_tensor, "1p", path_results_1 + "/mean_and_std_tensors_plots")
##input_tensor = input_tensor[0]
##CNN_model = CompletionN()
##path_job = "results_job_2025-02-05 16:58:54.736919"  
##path_job_2 = "results_job_2025-01-28 08:55:13.496992"
##list_masks = load_land_sea_masks("dataset_training/land_sea_masks/")
##path_mean_std = "results_job_2025-02-05 16:58:54.736919/results_training_1/mean_and_std_tensors"
##path_fig_channel = "results_job_2025-02-05 16:58:54.736919/results_training_1/P_l/200/lrc_0.001/plots"
##path_fig_channel_prof = "results_job_2025-02-05 16:58:54.736919/results_training_1/P_l/200/lrc_0.001/plots_prof_trial"
##path_fig_channel_prof_2 = "results_job_2025-01-28 08:55:13.496992/results_training_2_ensemble/P_l/20/lrc_0.001/plots_prof_2_trial"

##tensor_output_num_model = load_old_total_tensor("dataset_training/old_total_dataset/", 0, [(2019, 24, 2)])
##tensor_output_num_model = torch.unsqueeze(load_old_total_tensor("dataset_training/old_total_dataset/", 0, [(2019, 24, 2)])[:, :, :-1, :, 1:-1][:, 6, :, :, :], 1)
##list_to_plot_coordinates = load_transp_lat_coordinates("dataset_training/total_dataset/P_l/2019/", [(24, 2)])[0]
##print("list to plot coord", list_to_plot_coordinates)


##plot_NN_maps(tensor_output_num_model, list_masks, "P_l", "/leonardo_work/OGS23_PRACE_IT_0/ttonelli/CNN_reconstruction_final_resolution/results_job_2025-02-24 14:47:53.409525")

##plot_NN_maps_final_1(input_tensor, CNN_model, path_job, list_masks, "P_l", path_mean_std, path_fig_channel)
#plot_NN_maps_final_2(input_tensor, CNN_model, path_job_2, list_masks, "P_l", path_mean_std, path_fig_channel, 10)
#plot_models_profiles_1(input_tensor, CNN_model, tensor_output_num_model, path_job, "P_l", path_mean_std, path_fig_channel_prof, list_to_plot_coordinates)
#dovrei testarlo con un tensore di input di float, ma non ce lo ho e calcolarlo viene un po' complicato
#plot_models_profiles_2(input_tensor, CNN_model, tensor_output_num_model, path_job_2, "P_l", path_mean_std, path_fig_channel_prof_2, list_to_plot_coordinates, 10)



#PLOT NN MAPS 1 PHASE
#preparing paths
path_job = "/leonardo_work/OGS23_PRACE_IT_0/ttonelli/CNN_reconstruction_final_resolution/results_job_2025-02-22 10:42:50.013434"
path_results_1 = path_job + "/results_training_1"
list_masks = load_land_sea_masks("dataset_training/land_sea_masks/")
path_mean_std = path_results_1 + "/mean_and_std_tensors_plots/mean_and_std_tensors"
path_fig_channel = path_results_1 + "/P_l/200/lrc_0.001/plots_final"
if not os.path.exists(path_fig_channel):
    os.makedirs(path_fig_channel)

#preparing input data
input_tensor = load_tensors("dataset_training/total_dataset/P_l/2019/", [(24, 2)])
input_tensor, mean_tensor, std_tensor = Normalization(input_tensor, "1p", path_results_1 + "/mean_and_std_tensors_plots")
input_tensor = input_tensor[0]

#prepare CNN
CNN_model = CompletionN()

#plot the map
plot_NN_maps_final_1(input_tensor, CNN_model, path_job, list_masks, "P_l", path_mean_std, path_fig_channel, mean_layer=True, list_layers=[0, 40, 80, 120, 180, 300])



#PLOT PROFILES 1 PHASE
#preparing paths
path_job = "/leonardo_work/OGS23_PRACE_IT_0/ttonelli/CNN_reconstruction_final_resolution/results_job_2025-02-22 10:42:50.013434"
path_results_1 = path_job + "/results_training_1"
list_masks = load_land_sea_masks("dataset_training/land_sea_masks/")
path_mean_std = path_results_1 + "/mean_and_std_tensors_plots/mean_and_std_tensors"
path_fig_channel_prof_1 = path_results_1 + "/P_l/200/lrc_0.001/profiles_1_final"
if not os.path.exists(path_fig_channel_prof_1):
    os.makedirs(path_fig_channel_prof_1)

#preparing input data
input_tensor = load_tensors("dataset_training/total_dataset/P_l/2019/", [(24, 2)])
input_tensor, mean_tensor, std_tensor = Normalization(input_tensor, "1p", path_results_1 + "/mean_and_std_tensors_plots")
input_tensor = input_tensor[0]

#prepare CNN
CNN_model = CompletionN()

#load BFM tensor
tensor_output_num_model = torch.unsqueeze(load_old_total_tensor("dataset_training/old_total_dataset/", 0, [(2019, 24, 2)])[:, :, :-1, :, 1:-1][:, 6, :, :, :], 1)

#load coordinates ot plot
list_to_plot_coordinates = load_transp_lat_coordinates("dataset_training/total_dataset/P_l/2019/", [(24, 2)])[0]

plot_models_profiles_1(input_tensor, CNN_model, tensor_output_num_model, path_job, "P_l", path_mean_std, path_fig_channel_prof_1, list_to_plot_coordinates)