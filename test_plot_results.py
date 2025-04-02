#In this script we recreate the plots for POST-PORCESSING (so with already trained models) 


import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns

from convolutional_network import CompletionN
from hyperparameter import *
from normalization import Normalization, tmp_Normalization
from utils_function import compute_profile_coordinates
from utils_generation_train_1p import read_list, write_list
from utils_training_1 import load_tensors, load_land_sea_masks, load_old_total_tensor, load_transp_lat_coordinates, re_load_float_input_data, re_load_float_input_data_external
from plot_results_final import plot_NN_maps_final_1, plot_NN_maps_final_2, plot_models_profiles_1, plot_models_profiles_2, plot_BFM_maps, plot_Hovmoller
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

prob_statement = "hovmoller_external"


if prob_statement == "maps_1":
    #PLOT NN MAPS 1 PHASE
    #preparing paths
    #path_job = "/leonardo_work/OGS23_PRACE_IT_0/ttonelli/CNN_reconstruction_final_resolution/results_job_2025-02-22 10:42:50.013434"
    path_job = "/leonardo_work/OGS23_PRACE_IT_0/ttonelli/CNN_reconstruction_final_resolution/results_job_2025-02-14 10:05:14.974986"
    path_results_1 = path_job + "/results_training_1"
    path_lr = path_results_1 + "/P_l/200/lrc_0.001"
    list_ywd_indexes = read_list(path_lr + "/ywd_indexes.txt")
    index_external_test = read_list(path_lr + "/index_external_testing.txt")
    print("years week indexes", list_ywd_indexes)
    print("index external testing", index_external_test)
    list_masks = load_land_sea_masks("dataset_training/land_sea_masks/")
    path_mean_std = path_results_1 + "/mean_and_std_tensors" #"/mean_and_std_tensors_plots/mean_and_std_tensors"
    path_fig_channel = path_results_1 + "/P_l/200/lrc_0.001/plots_1_final"
    if not os.path.exists(path_fig_channel):
        os.makedirs(path_fig_channel)
    year_week_data = (2022, 4)
    path_single_data = path_fig_channel + "/year_" + str(year_week_data[0]) + "_week_" + str(year_week_data[1])
    if not os.path.exists(path_single_data):
        os.makedirs(path_single_data)
    path_fig_channel_CNN_maps = path_single_data + "/CNN_1_maps_final_resc_depth_high_res"
    if not os.path.exists(path_fig_channel_CNN_maps):
        os.makedirs(path_fig_channel_CNN_maps)
    path_fig_channel_BFM_maps = path_single_data + "/BFM_1_maps_final_resc_depth_high_res"
    if not os.path.exists(path_fig_channel_BFM_maps):
        os.makedirs(path_fig_channel_BFM_maps)
    for i in range(len(list_ywd_indexes)):
        if year_week_data[0] in list_ywd_indexes[i] and year_week_data[1] in list_ywd_indexes[i]:
            print("i", i)
            print("ywd", list_ywd_indexes[i])
            if i in index_external_test:
                print("found index", list_ywd_indexes[i][2])
                duplicate = list_ywd_indexes[i][2]
                break

    #preparing input data
    input_tensor = load_tensors("dataset_training/total_dataset/P_l/2022/", [(4, duplicate)])
    #input_tensor, mean_tensor, std_tensor = Normalization(input_tensor, "1p", path_results_1 + "/mean_and_std_tensors_plots")
    input_tensor = tmp_Normalization(input_tensor, "1p", path_mean_std)   
    input_tensor = input_tensor[0]

    #load BFM tensor
    tensor_BFM = torch.unsqueeze(load_old_total_tensor("dataset_training/old_total_dataset/", 0, [(2022, 4, duplicate)])[:, :, :-1, :, 1:-1][:, 6, :, :, :], 1)

    #prepare CNN
    CNN_model = CompletionN()

    #plot the map
    plot_NN_maps_final_1(input_tensor, CNN_model, path_job, list_masks, "P_l", path_mean_std, path_fig_channel_CNN_maps, mean_layer=True, list_layers=[0, 30, 60, 130, 180, 300])

    #plot the BFM map
    plot_BFM_maps(tensor_BFM, list_masks, "P_l", path_fig_channel_BFM_maps, [0, 30, 60, 130, 180, 300])


elif prob_statement == "prof_1":
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
    input_tensor = load_tensors("dataset_training/total_dataset/P_l/2019/", [(24, 26)])
    input_tensor, mean_tensor, std_tensor = Normalization(input_tensor, "1p", path_results_1 + "/mean_and_std_tensors_plots")
    input_tensor = input_tensor[0]

    #prepare CNN
    CNN_model = CompletionN()

    #load BFM tensor
    tensor_output_num_model = torch.unsqueeze(load_old_total_tensor("dataset_training/old_total_dataset/", 0, [(2019, 24, 26)])[:, :, :-1, :, 1:-1][:, 6, :, :, :], 1)

    #load coordinates ot plot
    list_to_plot_coordinates = load_transp_lat_coordinates("dataset_training/total_dataset/P_l/2019/", [(24, 26)])[0]

    #plot_models_profiles_1(input_tensor, CNN_model, tensor_output_num_model, path_job, "P_l", path_mean_std, path_fig_channel_prof_1, list_to_plot_coordinates)



elif prob_statement == "maps_2":
    #PLOT NN MAPS PHASE 2
    #preparing paths
    path_job = "/leonardo_work/OGS23_PRACE_IT_0/ttonelli/CNN_reconstruction_final_resolution/results_job_2025-02-14 10:05:14.974986"
    path_results_2 = path_job + "/results_training_2_ensemble"
    list_masks = load_land_sea_masks("dataset_training/land_sea_masks/")
    path_mean_std_2 = path_results_2 + "/mean_and_std_tensors"
    path_fig_channel_2 = path_results_2 + "/P_l/20/lrc_0.001/plots_2_final"
    if not os.path.exists(path_fig_channel_2):
        os.makedirs(path_fig_channel_2)
    year_week_data = (2021, 23)
    path_single_data_2 = path_fig_channel_2 + "/year_" + str(year_week_data[0]) + "_week_" + str(year_week_data[1])
    if not os.path.exists(path_single_data_2):
        os.makedirs(path_single_data_2)
    path_fig_channel_2_mean = path_single_data_2 + "/mean"
    if not os.path.exists(path_fig_channel_2_mean):
        os.makedirs(path_fig_channel_2_mean)
    path_fig_channel_2_std = path_single_data_2 + "/std"
    if not os.path.exists(path_fig_channel_2_std):
        os.makedirs(path_fig_channel_2_std)

    #preparing input data
    input_tensor_2 = re_load_float_input_data("/leonardo_work/OGS23_PRACE_IT_0/ttonelli/CNN_reconstruction_final_resolution/dataset", [(2021, 23)])
    input_tensor_2 = tmp_Normalization(input_tensor_2, "2p", path_mean_std_2)
    input_tensor_2 = input_tensor_2[0]

    plot_NN_maps_final_2(input_tensor_2, path_job, list_masks, "P_l", [path_fig_channel_2_mean, path_fig_channel_2_std], 10, mean_layer=True, list_layers = [0, 40, 80, 120, 140, 180, 300])


elif prob_statement == "prof_2":
    #PLOT PROFILES 2 PHASE
    #preparing paths
    path_job = "/leonardo_work/OGS23_PRACE_IT_0/ttonelli/CNN_reconstruction_final_resolution/results_job_2025-02-14 10:05:14.974986"
    path_results_2 = path_job + "/results_training_2_ensemble"
    list_masks = load_land_sea_masks("dataset_training/land_sea_masks/")
    path_mean_std_2 = path_results_2 + "/mean_and_std_tensors"
    path_fig_channel_2 = path_results_2 + "/P_l/20/lrc_0.001/plots_2_final"
    if not os.path.exists(path_fig_channel_2):
        os.makedirs(path_fig_channel_2)
    year_week_data = (2021, 4)
    path_single_data_2 = path_fig_channel_2 + "/year_" + str(year_week_data[0]) + "_week_" + str(year_week_data[1])
    if not os.path.exists(path_single_data_2):
        os.makedirs(path_single_data_2)
    path_fig_channel_prof_2 = path_single_data_2 + "/profiles_2_FINAL_resc_depth_high_res"
    if not os.path.exists(path_fig_channel_prof_2):
        os.makedirs(path_fig_channel_prof_2)

    #preparing input data
    #BFM input data
    input_tensor_BFM = torch.unsqueeze(load_old_total_tensor("dataset_training/old_total_dataset/", 0, [(2021, 4, 2)])[:, :, :-1, :, 1:-1][:, 6, :, :, :], 1)
    #input phys + float data
    input_tensor_2 = re_load_float_input_data("/leonardo_work/OGS23_PRACE_IT_0/ttonelli/CNN_reconstruction_final_resolution/dataset", [(2021, 4)])
    input_tensor_2 = tmp_Normalization(input_tensor_2, "2p", path_mean_std_2)
    input_tensor_2 = input_tensor_2[0]

    #list float profiles coordinates
    float_tensor = torch.load("/leonardo_work/OGS23_PRACE_IT_0/ttonelli/CNN_reconstruction_final_resolution/dataset/float/2021/final_tensor/P_l/datetime_4.pt")[:, :, :-2, :, 1:-1]
    list_float_profiles_coordinates = compute_profile_coordinates(float_tensor[:, 0:1, :, :, :])

    plot_models_profiles_2(input_tensor_2, input_tensor_BFM, float_tensor, path_job, "P_l", path_mean_std_2, path_fig_channel_prof_2, list_float_profiles_coordinates, 10)


elif prob_statement == "hovmoller":
    print(np.arange(0, 157, 13), flush=True)
    path_job = "/leonardo_work/OGS23_PRACE_IT_0/ttonelli/CNN_reconstruction_final_resolution/results_job_2025-02-14 10:05:14.974986"
    path_results_2 = path_job + "/results_training_2_ensemble"
    list_masks = load_land_sea_masks("dataset_training/land_sea_masks/")
    path_mean_std_2 = path_results_2 + "/mean_and_std_tensors"
    path_fig_channel_2 = path_results_2 + "/P_l/20/lrc_0.001/plots_2_final"
    if not os.path.exists(path_fig_channel_2):
        os.makedirs(path_fig_channel_2)
    list_yw_2019 = [(2019, i) for i in range(1, 53)]
    list_yw_2020 = [(2020, i) for i in range(1, 54)]
    list_yw_2021 = [(2021, i) for i in range(1, 53)]
    list_week_tensors_2019 = re_load_float_input_data("/leonardo_work/OGS23_PRACE_IT_0/ttonelli/CNN_reconstruction_final_resolution/dataset", list_yw_2019)
    list_week_tensors_2020 = re_load_float_input_data("/leonardo_work/OGS23_PRACE_IT_0/ttonelli/CNN_reconstruction_final_resolution/dataset", list_yw_2020)
    list_week_tensors_2021 = re_load_float_input_data("/leonardo_work/OGS23_PRACE_IT_0/ttonelli/CNN_reconstruction_final_resolution/dataset", list_yw_2021)
    print("list 2019", len(list_week_tensors_2019), flush=True)
    print("list 2020", len(list_week_tensors_2020), flush=True)
    print("list 2021", len(list_week_tensors_2021), flush=True)
    list_week_tensors = list_week_tensors_2019 + list_week_tensors_2020 + list_week_tensors_2021
    del list_week_tensors_2019
    del list_week_tensors_2020
    del list_week_tensors_2021
    print("list week tensors", len(list_week_tensors), flush=True)
    list_week_tensors_norm = tmp_Normalization(list_week_tensors, "2p", path_mean_std_2)
    list_week_tensors_BFM = [torch.unsqueeze(load_old_total_tensor("dataset_training/old_total_dataset/", 0, [(2019, i, 2)])[:, :, :-1, :, 1:-1][:, 6, :, :, :], 1) for i in range(1, 53)]

    plot_Hovmoller(list_week_tensors_norm, list_week_tensors_BFM, path_job, list_masks, "P_l", path_fig_channel_2, 1, "NWM", dict_coord_ga = {"NWM":[139, 153], "SWM": [111, 111], "TYR":[208, 124], "ION":[291, 69], "LEV":[444, 48]}, mean_layer=False, list_layers = [], mean_ga="mean_ngh")
    #plot_Hovmoller(list_week_tensors_norm, list_week_tensors_BFM, path_job, list_masks, "P_l", path_fig_channel_2, 1, "SWM", dict_coord_ga = {"NWM":[139, 153], "SWM": [111, 111], "TYR":[208, 124], "ION":[291, 69], "LEV":[444, 48]}, mean_layer=False, list_layers = [], mean_ga="mean_ngh")
    #plot_Hovmoller(list_week_tensors_norm, list_week_tensors_BFM, path_job, list_masks, "P_l", path_fig_channel_2, 1, "TYR", dict_coord_ga = {"NWM":[139, 153], "SWM": [111, 111], "TYR":[208, 124], "ION":[291, 69], "LEV":[444, 48]}, mean_layer=False, list_layers = [], mean_ga="mean_ngh")
    #plot_Hovmoller(list_week_tensors_norm, list_week_tensors_BFM, path_job, list_masks, "P_l", path_fig_channel_2, 1, "ION", dict_coord_ga = {"NWM":[139, 153], "SWM": [111, 111], "TYR":[208, 124], "ION":[291, 69], "LEV":[444, 48]}, mean_layer=False, list_layers = [], mean_ga="mean_ngh")
    plot_Hovmoller(list_week_tensors_norm, list_week_tensors_BFM, path_job, list_masks, "P_l", path_fig_channel_2, 1, "LEV", dict_coord_ga = {"NWM":[139, 153], "SWM": [111, 111], "TYR":[208, 124], "ION":[291, 69], "LEV":[444, 48]}, mean_layer=False, list_layers = [], mean_ga="mean_ngh")


elif prob_statement == "hovmoller_external":
    print("check torch equality")
    file_2 = torch.load("/leonardo_work/OGS23_PRACE_IT_0/ttonelli/CNN_reconstruction_final_resolution/dataset/float/2019/final_tensor/P_l/datetime_14.pt")
    file_1 = torch.load("/leonardo_work/OGS23_PRACE_IT_0/ttonelli/CNN_reconstruction_final_resolution/dataset/float/2019/interp_tensor/P_l/datetime_2019_14.pt")
    file_3 = torch.load("/leonardo_work/OGS23_PRACE_IT_0/ttonelli/CNN_reconstruction_final_resolution/dataset/float/2019/final_tensor/P_l/datetime_24.pt")
    file_4 = torch.load("/leonardo_work/OGS23_PRACE_IT_0/ttonelli/CNN_reconstruction_final_resolution/dataset/float/2019/interp_tensor/P_l/datetime_2019_24.pt")
    file_5 = torch.load("/leonardo_work/OGS23_PRACE_IT_0/ttonelli/CNN_reconstruction_final_resolution/dataset/float/2019/final_tensor/P_l/datetime_38.pt")
    file_6 = torch.load("/leonardo_work/OGS23_PRACE_IT_0/ttonelli/CNN_reconstruction_final_resolution/dataset/float/2019/interp_tensor/P_l/datetime_2019_38.pt")
    print(torch.equal(file_1, file_2))
    print(torch.equal(file_4, file_3))
    print(torch.equal(file_5, file_6))
    print(torch.equal(file_1, file_3))
    path_job = "/leonardo_work/OGS23_PRACE_IT_0/ttonelli/CNN_reconstruction_final_resolution/results_job_2025-02-14 10:05:14.974986"
    path_results_2 = path_job + "/results_training_2_ensemble"
    list_masks = load_land_sea_masks("dataset_training/land_sea_masks/")
    path_mean_std_2 = path_results_2 + "/mean_and_std_tensors"
    path_fig_channel_2 = path_results_2 + "/P_l/20/lrc_0.001/plots_2_final"
    if not os.path.exists(path_fig_channel_2):
        os.makedirs(path_fig_channel_2)
    list_wd_2022_phys = [(2022, i) for i in range(1, 53)]
    list_yw_2023 = [(2023, i) for i in range(1, 53)]
    list_week_tensors_2023 = re_load_float_input_data_external("/leonardo_work/OGS23_PRACE_IT_0/ttonelli/CNN_reconstruction_final_resolution/dataset", list_yw_2023, list_wd_2022_phys)
    print("list 2023", len(list_week_tensors_2023), flush=True)
    list_week_tensors = list_week_tensors_2023 
    del list_week_tensors_2023
    print("list week tensors", len(list_week_tensors), flush=True)
    list_week_tensors_norm = tmp_Normalization(list_week_tensors, "2p", path_mean_std_2)
    list_week_tensors_BFM = [torch.unsqueeze(load_old_total_tensor("dataset_training/old_total_dataset/", 0, [(2019, i, 2)])[:, :, :-1, :, 1:-1][:, 6, :, :, :], 1) for i in range(1, 53)]
    print(np.arange(0, len(list_week_tensors), 13), flush=True)
    list_week_tensors_norm_reverse = list_week_tensors_norm[40:] + list_week_tensors_norm[:40]
    print("len reverse list", len(list_week_tensors_norm_reverse))

    #plot_Hovmoller(list_week_tensors_norm, list_week_tensors_BFM, path_job, list_masks, "P_l", path_fig_channel_2, 1, "NWM", dict_coord_ga = {"NWM":[139, 153], "SWM": [111, 111], "TYR":[208, 124], "ION":[291, 69], "LEV":[444, 48]}, mean_layer=False, list_layers = [], mean_ga="mean_ngh")
    #plot_Hovmoller(list_week_tensors_norm, list_week_tensors_BFM, path_job, list_masks, "P_l", path_fig_channel_2, 1, "SWM", dict_coord_ga = {"NWM":[139, 153], "SWM": [111, 111], "TYR":[208, 124], "ION":[291, 69], "LEV":[444, 48]}, mean_layer=False, list_layers = [], mean_ga="mean_ngh")
    #plot_Hovmoller(list_week_tensors_norm, list_week_tensors_BFM, path_job, list_masks, "P_l", path_fig_channel_2, 1, "TYR", dict_coord_ga = {"NWM":[139, 153], "SWM": [111, 111], "TYR":[208, 124], "ION":[291, 69], "LEV":[444, 48]}, mean_layer=False, list_layers = [], mean_ga="mean_ngh")
    #plot_Hovmoller(list_week_tensors_norm, list_week_tensors_BFM, path_job, list_masks, "P_l", path_fig_channel_2, 1, "ION", dict_coord_ga = {"NWM":[139, 153], "SWM": [111, 111], "TYR":[208, 124], "ION":[291, 69], "LEV":[444, 48]}, mean_layer=False, list_layers = [], mean_ga="mean_ngh")
    plot_Hovmoller(list_week_tensors_norm_reverse, list_week_tensors_BFM, path_job, list_masks, "P_l", path_fig_channel_2, 1, "LEV", dict_coord_ga = {"NWM":[139, 153], "SWM": [111, 111], "TYR":[208, 124], "ION":[291, 69], "LEV":[444, 48]}, mean_layer=False, list_layers = [], mean_ga="mean_ngh")