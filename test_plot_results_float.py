#In this script we recreate the plots for POST-PORCESSING (so with already trained models) 


import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns

from convolutional_network import CompletionN
from hyperparameter import *
from normalization import Normalization, tmp_Normalization, tmp_Normalization_float
from utils_function import compute_profile_coordinates
from utils_generation_train_1p import read_list, write_list
from utils_training_1 import load_tensors, load_land_sea_masks, load_old_total_tensor, load_transp_lat_coordinates, re_load_float_input_data, re_load_float_input_data_external
from plot_results_final import plot_NN_maps_final_1, plot_NN_maps_final_2, plot_models_profiles_1, plot_models_profiles_2, plot_BFM_maps, plot_Hovmoller, plot_Hovmoller_real_float
from plot_results import plot_NN_maps


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

sns.set_theme(context='paper', style='whitegrid', font='sans-serif', font_scale=1.5,
              color_codes=True, rc=None)


prob_statement = "hovmoller_float"

if prob_statement == "hovmoller_float":
    total_float_tensor = torch.load("total_float_tensor.pt")
    #path_images = "float_plots/daily_total/"
    #if not os.path.exists(path_images):
    #    os.makedirs(path_images)
    #for i in range(107):
    #    plt.plot(torch.squeeze(total_float_tensor[:, i]).flip(0), np.arange(0, 31), linewidth=2.0)
    #    plt.savefig(path_images + str(i) + ".png", dpi=100)
    #    plt.close()
    weekly_total_float_tensor = torch.load("weekly_mean_float_tensor.pt")
    path_images = "float_plots/weekly_total/"
    if not os.path.exists(path_images):
        os.makedirs(path_images)
    for i in range(250, 280):
        plt.plot(torch.squeeze(weekly_total_float_tensor[:, i]).flip(0), np.arange(0, 31), linewidth=2.0)
        plt.savefig(path_images + str(i) + ".png", dpi=100)
        plt.close()
    path_job = "/leonardo_work/OGS23_PRACE_IT_0/ttonelli/CNN_reconstruction_final_resolution/results_job_2025-02-14 10:05:14.974986"
    path_results_2 = path_job + "/results_training_2_ensemble"
    list_masks = load_land_sea_masks("dataset_training/land_sea_masks/")
    path_mean_std_2 = path_results_2 + "/mean_and_std_tensors"
    path_fig_channel_2 = path_results_2 + "/P_l/20/lrc_0.001/plots_2_final"
    if not os.path.exists(path_fig_channel_2):
        os.makedirs(path_fig_channel_2)
    float_device_tensor_LEV = torch.load("weekly_LEV_mean_float_tensor_no_0.pt")
    print(float_device_tensor_LEV.shape)
    print(torch.equal(float_device_tensor_LEV, weekly_total_float_tensor[:, :58]))
    #float_device_tensor_LEV = tmp_Normalization_float(float_device_tensor_LEV, "2p", path_mean_std_2, [444, 48])
    path_images = "float_plots/weekly_LEV_total/"
    #if not os.path.exists(path_images):
    #    os.makedirs(path_images)
    #for i in range(float_device_tensor_LEV.shape[1]):
    #    plt.plot(torch.squeeze(float_device_tensor_LEV[:, i]).flip(0), np.arange(0, 31), linewidth=2.0)
    #    plt.savefig(path_images + str(i) + ".png", dpi=100)
    #    plt.close()
    float_device_tensor_NWM_5 = torch.load("weekly_NWM_5906990_mean_float_tensor_no_0.pt")
    float_device_tensor_NWM_1 = torch.load("weekly_NWM_1902605_mean_float_tensor.pt")
    print("shape 5", float_device_tensor_NWM_5.shape)
    print(torch.equal(float_device_tensor_NWM_5, weekly_total_float_tensor[:, 422:492]))
    print("shape 1", float_device_tensor_NWM_1.shape)
    print(torch.equal(float_device_tensor_NWM_1, weekly_total_float_tensor[:, 247:422]))
    flipped_LEV = float_device_tensor_LEV.flip(0)
    print(flipped_LEV.shape)
    print("no flip", float_device_tensor_LEV[:, 13])
    print("flip", flipped_LEV[:, 13])
    weekly_total_float_tensor_no_0 = torch.load("weekly_mean_float_tensor_no_0.pt")
    #float_device_tensor_NWM = tmp_Normalization_float(float_device_tensor_NWM, "2p", path_mean_std_2, [139, 153])
    #plot_Hovmoller_real_float(weekly_total_float_tensor_no_0, path_fig_channel_2, "MED", mean_layer=False, list_layers = [], mean_ga=True, tensor_order="standard", name_fig="weekly_total_no_0")
    #plot_Hovmoller_real_float(float_device_tensor_LEV, path_fig_channel_2, "LEV", mean_layer=False, list_layers = [], mean_ga=True, tensor_order="LEV_order", name_fig="weekly_no_0_final_sept_aug", apply_prof_smooting=True)
    plot_Hovmoller_real_float(weekly_total_float_tensor_no_0, path_fig_channel_2, "NWM", mean_layer=False, list_layers = [], mean_ga=True, tensor_order="NWM_order", name_fig="weekly_no_0_final_oct_sept_ls", apply_prof_smooting=True)