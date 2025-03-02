#In this script we test the rmse function, settings all the inputs that this function requires

import numpy as np
import torch 
import os 

from convolutional_network import CompletionN
from hyperparameter import *
from normalization import Normalization, tmp_Normalization
from rmse_functions import select_season_tensors, create_ga_mask_full, compute_rmse_ga_season_2, compute_rmse_ga_season_2_ensemble, rmse_ga_season_2, RMSE_ensemble_ga_season#
from utils_function import compute_profile_coordinates
from utils_generation_train_1p import write_list, read_list
from utils_training_1 import load_land_sea_masks, re_load_tensors, re_load_old_float_tensors, re_load_float_input_data


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device", device)

#a --> test for RMSE function for 1 season and 1 ga
#path_job = "/leonardo_work/OGS23_PRACE_IT_0/ttonelli/CNN_reconstruction_final_resolution/results_job_2025-01-30 11:34:44.708444"
#years_week_duplicates_indexes = read_list(path_job + "/results_training_1/P_l/200/lrc_0.001" + "/ywd_indexes.txt")[:20]
#print("len years week indexes", years_week_duplicates_indexes, flush=True)
#list_tensors = re_load_tensors("dataset_training/total_dataset", years_week_duplicates_indexes[:20])
#print("len list tensors", len(list_tensors), flush=True)
#season = "winter"
#sst = select_season_tensors(list_tensors, season, years_week_duplicates_indexes)
#print("sst", len(sst[0]), flush=True)

#my_ga = "NWM"
#list_float_tensors = re_load_old_float_tensors("/leonardo_work/OGS23_PRACE_IT_0/ttonelli/CNN_reconstruction_final_resolution/dataset", years_week_duplicates_indexes)
#print("float tensor shape", list_float_tensors[0].shape)
#list_float_coordinates = [compute_profile_coordinates(torch.unsqueeze(float_tensor[:, -1, :, :, :], 1)) for float_tensor in list_float_tensors]
#print("single float coordinate", list_float_coordinates[1])
#tensor_shape = (1, 1, d, h, w)
#ga_masks = [create_ga_mask(my_ga, list_float_coordinates[i], tensor_shape) for i in range(len(list_float_coordinates))]
#print("mask shape", ga_masks[0].shape, flush=True)
#print("profiles counter", torch.count_nonzero(list_float_tensors[0][:, -1, :, :, :]), flush=True)
#print("ga mask counter", torch.count_nonzero(ga_masks[0]), flush=True)


#list_tensors, _, _ = Normalization(list_tensors, "1p", path_job + "/results_training_1/mean_and_std_tensors")
#list_float_tensors = [float_tensor[:, :, :-1, :, :] for float_tensor in list_float_tensors]
#my_mean = torch.unsqueeze(torch.load(path_job + "/results_training_2_ensemble/mean_and_std_tensors" + "/mean_tensor.pt")[:, 6, :, :, :], 1).to(device)
#my_std = torch.unsqueeze(torch.load(path_job + "/results_training_2_ensemble/mean_and_std_tensors" + "/std_tensor.pt")[:, 6, :, :, :], 1).to(device)
#land_sea_masks = load_land_sea_masks("dataset_training/land_sea_masks/")
#exp_weights = torch.ones([1, 1, d-2, h, w-2])
#CNN_model = CompletionN()
#checkpoint_CNN = torch.load(path_job + "/results_training_2_ensemble/P_l/20/lrc_0.001/ensemble_model_1/" + "/" + 'model_checkpoint_2_ens_1.pth', map_location=device)
#CNN_model.load_state_dict(checkpoint_CNN['model_state_dict'])
#compute_rmse_ga_season_2(list_tensors, list_float_tensors, list_float_coordinates, land_sea_masks, my_ga, season, years_week_duplicates_indexes, exp_weights, CNN_model, my_mean, my_std)




#OLD APPROACH --> it all works, but input data are not properly correct and the GA mask is applied before the training of teh networkk
#path_job = "/leonardo_work/OGS23_PRACE_IT_0/ttonelli/CNN_reconstruction_final_resolution/results_job_2025-01-30 11:34:44.708444"
#years_week_duplicates_indexes = read_list(path_job + "/results_training_1/P_l/200/lrc_0.001" + "/ywd_indexes.txt")[:20]
#print("list year week duplicates indexes", years_week_duplicates_indexes)
#list_tensors = re_load_tensors("dataset_training/total_dataset", years_week_duplicates_indexes[:20])
#list_float_tensors = re_load_old_float_tensors("/leonardo_work/OGS23_PRACE_IT_0/ttonelli/CNN_reconstruction_final_resolution/dataset", years_week_duplicates_indexes)
#list_float_coordinates = [compute_profile_coordinates(torch.unsqueeze(float_tensor[:, -1, :, :, :], 1)) for float_tensor in list_float_tensors]
#list_tensors, _, _ = Normalization(list_tensors, "1p", path_job + "/results_training_1/mean_and_std_tensors")
#list_float_tensors = [float_tensor[:, :, :-1, :, :] for float_tensor in list_float_tensors]
#path_mean_std = path_job + "/results_training_2_ensemble/mean_and_std_tensors"
#exp_weights = torch.ones([1, 1, d-2, h, w-2])

#ensemble_mode = False
#if ensemble_mode == False:
#    CNN_model = CompletionN()
#    checkpoint_CNN = torch.load(path_job + "/results_training_2/model_checkpoint_2.pth", map_location=device)
#    CNN_model.load_state_dict(checkpoint_CNN['model_state_dict'])
#    #without ensemble --> 1 single CNN model
#    loss_results = []
#    for my_ga in ["TYR"]:  #list(dict_ga.keys()):
#        for season in ["winter"]:   #list(dict_season.keys()):
#            list_loss_ga_season = []
#            list_loss_ga_season.append(my_ga)
#            list_loss_ga_season.append(season)
#            print("my ga = ", my_ga)
#            print("my season = ", season)
#            loss_ga_season = compute_rmse_ga_season_2(list_tensors, list_float_tensors, list_float_coordinates, my_ga, season, years_week_indexes, exp_weights, CNN_model, path_mean_std)
#            list_loss_ga_season.append(loss_ga_season)
#            loss_results.append(list_loss_ga_season)
#else:
#    list_CNN_model = [CompletionN() for i in range(10)]
#    list_checkpoints_CNN = [torch.load(path_job + "/results_training_2_ensemble/P_l/20/lrc_0.001/ensemble_model_" + str(i_ens) + "/" + 'model_checkpoint_2_ens_' + str(i_ens) + '.pth', map_location=device) for i_ens in range(10)]
#    for i_ens in range(len(list_CNN_model)):
#        list_CNN_model[i_ens].load_state_dict(list_checkpoints_CNN[i_ens]['model_state_dict'])
#    loss_results = []
#    for my_ga in list(dict_ga.keys()):
#        for season in ["winter"]:   #list(dict_season.keys()):
#            list_loss_ga_season = []
#            list_loss_ga_season.append(my_ga)
#            list_loss_ga_season.append(season)
#            print("my ga = ", my_ga)
#            print("my season = ", season)
#            loss_ga_season = compute_rmse_ga_season_2_ensemble(list_tensors, list_float_tensors, list_float_coordinates, my_ga, season, years_week_indexes, exp_weights, list_CNN_model, path_mean_std)
#            list_loss_ga_season.append(loss_ga_season)
#            loss_results.append(list_loss_ga_season)

##write the list of losses in a txt file
#for list_loss in loss_results:
#    with open(path_job + "/results_training_2_ensemble/" + "losses_ga_season.txt", "a") as f:
#        f.write(",".join([list_loss[0], list_loss[1], str(list_loss[2])]))




path_job = "/leonardo_work/OGS23_PRACE_IT_0/ttonelli/CNN_reconstruction_final_resolution/results_job_2025-02-22 10:42:50.013434"
path_mean_std = path_job + "/results_training_2/mean_and_std_tensors"
years_week_indexes = read_list(path_job + "/results_training_2/P_l/40/lrc_0.001" + "/ywd_indexes.txt")
print("list year week duplicates indexes", years_week_indexes)
index_training_2 = read_list(path_job + "/results_training_2/P_l/40/lrc_0.001" + "/index_training.txt")
index_int_testing_2 = read_list(path_job + "/results_training_2/P_l/40/lrc_0.001" + "/index_internal_testing.txt")
index_ext_testing_2 = read_list(path_job + "/results_training_2/P_l/40/lrc_0.001" + "/index_external_testing.txt")

#float dataset
list_float_tensors = re_load_old_float_tensors("/leonardo_work/OGS23_PRACE_IT_0/ttonelli/CNN_reconstruction_final_resolution/dataset", years_week_indexes)
list_float_coordinates = [compute_profile_coordinates(torch.unsqueeze(float_tensor[:, -1, :, :, :], 1)) for float_tensor in list_float_tensors]
list_float_tensors = [float_tensor[:, :, :-1, :, :] for float_tensor in list_float_tensors]

#total input dataset phase 2
input_dataset_2 = re_load_float_input_data("/leonardo_work/OGS23_PRACE_IT_0/ttonelli/CNN_reconstruction_final_resolution/dataset", years_week_indexes)
input_dataset_2 = tmp_Normalization(input_dataset_2, "2p", path_mean_std)

#train and test dataset phase 2
train_dataset_2 = [input_dataset_2[i] for i in index_training_2]
test_dataset_2 = [input_dataset_2[i] for i in index_ext_testing_2] + [input_dataset_2[i] for i in index_int_testing_2]                     



ensemble_mode = "no_false_no_true"
if ensemble_mode == False:
    CNN_model = CompletionN()
    checkpoint_CNN = torch.load(path_job + "/results_training_2/model_checkpoint_2.pth", map_location=device)
    CNN_model.load_state_dict(checkpoint_CNN['model_state_dict'])
    #without ensemble --> 1 single CNN model
    loss_results = []
    for my_ga in ["NWM"]:  #list(dict_ga.keys()):
        for season in ["winter"]:   #list(dict_season.keys()):
            list_loss_ga_season = []
            list_loss_ga_season.append(my_ga)
            list_loss_ga_season.append(season)
            print("my ga = ", my_ga)
            print("my season = ", season)
            loss_ga_season_train, loss_ga_season_test = rmse_ga_season_2(input_dataset_2, index_training_2, index_ext_testing_2 + index_int_testing_2, list_float_tensors, list_float_coordinates, CNN_model, years_week_indexes, my_ga, season, path_mean_std)
            list_loss_ga_season.append([loss_ga_season_train, loss_ga_season_test])
            loss_results.append(list_loss_ga_season)
elif ensemble_mode == True:
    #questa parte non è stata ricontrollata, ma basta usare la funzione dentro a rmse_functions.py, quella per gli ensemble
    list_CNN_model = [CompletionN() for i in range(10)]
    path_lr = path_job + "/results_training_2_ensemble/P_l/20/lrc_0.001"
    list_checkpoints_CNN = [torch.load(path_lr + "/ensemble_model_" + str(i_ens) + "/" + 'model_checkpoint_2_ens_' + str(i_ens) + '.pth', map_location=device) for i_ens in range(10)]
    for i_ens in range(len(list_CNN_model)):
        list_CNN_model[i_ens].load_state_dict(list_checkpoints_CNN[i_ens]['model_state_dict'])
    loss_results = []
    for my_ga in list(dict_ga.keys()):
        for season in ["winter"]:   #list(dict_season.keys()):
            list_loss_ga_season = []
            list_loss_ga_season.append(my_ga)
            list_loss_ga_season.append(season)
            print("my ga = ", my_ga)
            print("my season = ", season)
            loss_ga_season = compute_rmse_ga_season_2_ensemble(input_dataset_2, list_float_tensors, list_float_coordinates, my_ga, season, years_week_indexes, torch.ones([1, 1, d-2, h, w-2]), list_CNN_model, path_mean_std)
            list_loss_ga_season.append(loss_ga_season)
            loss_results.append(list_loss_ga_season)

#write the list of losses in a txt file
#for list_loss in loss_results:
#    with open(path_job + "/results_training_2/" + "losses_ga_season.txt", "a") as f:
#        f.write("\n")
#        f.write(",".join([list_loss[0], list_loss[1], str(list_loss[2])]) + "\n")





RMSE_ensemble_ga_season(path_job, years_week_indexes, 2)
